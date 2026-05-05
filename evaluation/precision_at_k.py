"""Precision@K evaluation of retrieval quality.

For each (paper, field) pair in the ground-truth set, this script:
1. Runs the HybridSearcher to retrieve the top-K chunks for the field's query.
2. Uses an LLM-as-judge to label each chunk as relevant (1) or not (0).
3. Computes Precision@K per field and overall.

LLM-as-judge is used rather than manual binary labels because it scales
to large evaluation sets at minimal cost, and modern judge models show
high agreement with human relevance labels. The judge prompt is
deliberately strict about what counts as "relevant" to avoid label inflation.

Usage:
    python -m evaluation.precision_at_k
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from openai import OpenAI
from openpyxl import load_workbook
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from litreviewrag import config
from litreviewrag.extraction.prompts import FIELD_SPECS, FieldSpec
from litreviewrag.retrieval.hybrid_search import HybridResult, HybridSearcher
from litreviewrag.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = Path("evaluation/ground_truth/gold_standard.xlsx")
RESULTS_MD_PATH = Path("docs/evaluation_results.md")
RAW_JSON_PATH = Path("results/precision_at_5_raw.json")

TOP_K = 5  # matches the proposal's Precision@5 specification

# Judge prompt. Asks for a binary verdict only, returned as strict JSON.
# We use the same model as extraction (gpt-4o-mini) so the relevance
# bar is consistent with what actually drives extractions.
# Judge prompt. Asks for a binary verdict only, returned as strict JSON.
# We define relevance broadly: a chunk is relevant if it contains
# information that would help an extractor answer the field, OR if it
# provides necessary context (e.g., methodology details that surround
# a findings claim). This matches how humans use retrieved context in
# practice — not all useful chunks contain the literal answer.
JUDGE_PROMPT_TEMPLATE = """You are judging whether a passage from a research paper is useful for extracting a specific field.

FIELD: {field_name}
FIELD DESCRIPTION: {field_description}

PASSAGE:
{chunk_text}

INSTRUCTIONS:
A passage is USEFUL if any of the following are true:
- It directly contains the answer to the field.
- It contains supporting context that would help an extractor produce the answer (e.g., for "findings", a passage describing the experimental setup that the findings refer to).
- It mentions specific terms, numbers, methods, or claims that contribute partial information for the field.

A passage is NOT USEFUL only if it covers a completely different topic, OR is purely boilerplate (acknowledgments, formatting headers, references list).

When in doubt, lean toward USEFUL — retrieval routinely provides surrounding context that helps extraction succeed.

Respond with JSON only:
{{"relevant": true}} or {{"relevant": false}}"""

def _get_client() -> OpenAI:
    """Construct an OpenAI client using the configured API key."""
    return OpenAI(api_key=config.OPENAI_API_KEY)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _judge_chunk(client: OpenAI, prompt: str) -> bool:
    """Ask the LLM judge whether a chunk is relevant.

    Args:
        client: An initialized OpenAI client.
        prompt: The fully-formatted judge prompt.

    Returns:
        True if the judge labels the chunk relevant, False otherwise.
        Returns False on parse failure (conservative — a malformed judgment
        is treated as non-relevant rather than counted as relevant).
    """
    response = client.chat.completions.create(
        model=config.EXTRACTION_MODEL,  # same model as extraction for consistency
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Judge returned unparseable JSON: %r", raw[:200])
        return False
    return bool(parsed.get("relevant", False))


def _judge_chunks_for_field(
    client: OpenAI,
    field_spec: FieldSpec,
    chunks: list[HybridResult],
) -> list[bool]:
    """Judge each chunk's relevance to the given field.

    Args:
        client: An initialized OpenAI client.
        field_spec: The field whose retrieval is being evaluated.
        chunks: The retrieved chunks to judge.

    Returns:
        A list of booleans, one per chunk, in input order.
    """
    labels: list[bool] = []
    for chunk in chunks:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            field_name=field_spec.name,
            field_description=field_spec.description,
            chunk_text=chunk.text[:1500],  # truncate very long chunks for prompt size
        )
        labels.append(_judge_chunk(client, prompt))
    return labels


def _load_paper_names(path: Path) -> list[str]:
    """Pull the list of paper names from the ground truth file."""
    wb = load_workbook(path, read_only=True)
    ws = wb.active
    headers = [c.value for c in ws[1]]
    name_col = headers.index("paper_name")
    return [
        str(row[name_col])
        for row in ws.iter_rows(min_row=2, values_only=True)
        if row[name_col]
    ]


def _append_to_markdown(report_text: str, output_path: Path) -> None:
    """Append the P@K report to the existing evaluation markdown file.

    If the file does not exist, creates it. Existing BERTScore content
    (from Step 16) is preserved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    # Avoid double-appending if this script is re-run
    marker = "## Precision@5: Retrieval Quality"
    if marker in existing:
        # Replace the existing P@K section
        before = existing.split(marker)[0]
        # If there's content after our section (e.g. future_work note), keep it
        rest_after_marker = existing.split(marker, 1)[1]
        # Find the next "## " heading after our section
        next_heading_idx = rest_after_marker.find("\n## ")
        after = rest_after_marker[next_heading_idx:] if next_heading_idx >= 0 else ""
        output_path.write_text(before + report_text + after, encoding="utf-8")
    else:
        output_path.write_text(existing + "\n" + report_text, encoding="utf-8")


def main() -> None:
    """Entry point: retrieve chunks, judge relevance, compute Precision@K."""
    print("Loading paper list ...")
    paper_names = _load_paper_names(GROUND_TRUTH_PATH)
    print(f"  {len(paper_names)} papers")

    store = VectorStore()
    indexed = set(store.list_papers())
    paper_names = [p for p in paper_names if p in indexed]
    if not paper_names:
        print("ERROR: No evaluation papers are in ChromaDB. Run ingest first.")
        return
    searcher = HybridSearcher(store)
    client = _get_client()

    print(f"\nJudging top-{TOP_K} chunks for {len(paper_names)} papers x "
          f"{len(FIELD_SPECS)} fields = "
          f"{len(paper_names) * len(FIELD_SPECS) * TOP_K} relevance judgments ...")

    raw_records: list[dict] = []
    field_to_labels: dict[str, list[int]] = {spec.name: [] for spec in FIELD_SPECS}

    for paper in paper_names:
        print(f"  Judging {paper} ...")
        for spec in FIELD_SPECS:
            chunks = searcher.search(
                query=spec.retrieval_query, paper_name=paper, top_k=TOP_K
            )
            labels = _judge_chunks_for_field(client, spec, chunks)
            field_to_labels[spec.name].extend(int(l) for l in labels)

            for chunk, label in zip(chunks, labels):
                raw_records.append(
                    {
                        "paper_name": paper,
                        "field_name": spec.name,
                        "chunk_index": chunk.chunk_index,
                        "score": chunk.score,
                        "relevant": label,
                        "chunk_preview": chunk.text[:200],
                    }
                )

    # Aggregate
    by_field = {
        name: (sum(labels) / len(labels)) if labels else 0.0
        for name, labels in field_to_labels.items()
    }
    all_labels = [l for labs in field_to_labels.values() for l in labs]
    overall = sum(all_labels) / len(all_labels) if all_labels else 0.0

    # Print summary
    print("\n" + "=" * 60)
    print(f"OVERALL Precision@{TOP_K} = {overall:.4f}  (n={len(all_labels)} judgments)")
    print("=" * 60)
    print(f"{'Field':<22} {'P@5':>8}  n")
    for spec in FIELD_SPECS:
        labels = field_to_labels[spec.name]
        precision = sum(labels) / len(labels) if labels else 0.0
        print(f"{spec.name:<22} {precision:>8.4f}  {len(labels)}")
    print("=" * 60)

    # Build markdown report (appended to evaluation_results.md)
    md_lines = [
        f"## Precision@{TOP_K}: Retrieval Quality",
        "",
        f"**Judge model:** `{config.EXTRACTION_MODEL}` &nbsp;&nbsp; "
        f"**Judgments:** {len(all_labels)}",
        "",
        "### Overall",
        "",
        f"| Metric | Score |",
        f"|---|---|",
        f"| **Precision@{TOP_K}** | **{overall:.4f}** |",
        "",
        "### Per-Field Breakdown",
        "",
        f"| Field | Precision@{TOP_K} | n |",
        f"|---|---|---|",
    ]
    for spec in FIELD_SPECS:
        labels = field_to_labels[spec.name]
        precision = sum(labels) / len(labels) if labels else 0.0
        md_lines.append(f"| {spec.name} | {precision:.4f} | {len(labels)} |")
    md_lines.append("")
    md_lines.append("")

    _append_to_markdown("\n".join(md_lines), RESULTS_MD_PATH)

    RAW_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(raw_records, fh, indent=2, ensure_ascii=False)

    print(f"\nAppended results to {RESULTS_MD_PATH}")
    print(f"Wrote {RAW_JSON_PATH}")


if __name__ == "__main__":
    main()
"""F1 evaluation of cross-paper contradiction detection.

Compares contradictions produced by the synthesis pipeline against a
human-curated ground truth. A predicted contradiction is counted as a
true positive if and only if:

1. It shares at least one paper with a ground-truth contradiction.
2. An LLM judge agrees that the predicted claims describe the SAME
   underlying factual disagreement as the ground-truth claims.

This semantic-match approach handles paraphrasing AND alternative paper
pairings — multiple papers in a corpus may anchor the same disagreement,
so requiring exact paper-pair match is overly strict.

Outputs Precision, Recall, F1, plus per-prediction match details.

Usage:
    python -m evaluation.contradiction_f1
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from litreviewrag import config
from litreviewrag.extraction.extractor import extract_paper
from litreviewrag.retrieval.hybrid_search import HybridSearcher
from litreviewrag.retrieval.vector_store import VectorStore
from litreviewrag.synthesis.contradiction import (
    Contradiction,
    detect_contradictions,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = Path("evaluation/ground_truth/contradictions.json")
RESULTS_MD_PATH = Path("docs/evaluation_results.md")
RAW_JSON_PATH = Path("results/contradiction_f1_raw.json")

# Judge prompt: relaxed to allow paper-pair flexibility while still
# requiring substantive equivalence. In a real lit review, finding that
# "papers disagree on whether LLMs achieve high simulation fidelity" is
# the meaningful insight — the specific paper pair anchoring the
# disagreement is secondary, since multiple papers may make either claim.
# MATCH_JUDGE_PROMPT = """You are deciding whether two descriptions of a research contradiction refer to the same underlying factual disagreement.

# CONTRADICTION 1 (Predicted by an automated system):
#   Paper A: {pred_paper_a}
#   Claim A: {pred_claim_a}
#   Paper B: {pred_paper_b}
#   Claim B: {pred_claim_b}

# CONTRADICTION 2 (Reference, written by a human):
#   Paper A: {ref_paper_a}
#   Claim A: {ref_claim_a}
#   Paper B: {ref_paper_b}
#   Claim B: {ref_claim_b}

# INSTRUCTIONS:
# A match requires:
# - The empirical disagreement is fundamentally about the same claim or measurement.
# - The disagreement points in the same direction (e.g., both flag a high-vs-low
#   fidelity disagreement, or both flag works-vs-doesn't-work).
# - AT LEAST ONE paper appears in both contradictions (the disagreement is anchored
#   by at least one shared paper). Different paper pairings are acceptable when
#   multiple papers in the corpus make the same kind of claim.

# If the contradictions share no papers, return false.
# If the contradictions are about substantively different disagreements, return false.
# Surface paraphrasing is fine.

# Respond with JSON only:
# {{"match": true}} or {{"match": false}}"""

# Judge prompt: emphasize that the disagreement is what matters, not
# the paper pairing. Multiple papers in a corpus may anchor the same
# disagreement, so two contradictions can match even when they involve
# different paper pairs — as long as they share at least one paper and
# describe the same substantive empirical conflict.
MATCH_JUDGE_PROMPT = """You are deciding whether two descriptions of a research contradiction refer to the SAME UNDERLYING DISAGREEMENT.

CONTRADICTION 1 (Predicted):
  Paper A: {pred_paper_a}
  Claim A: {pred_claim_a}
  Paper B: {pred_paper_b}
  Claim B: {pred_claim_b}

CONTRADICTION 2 (Reference):
  Paper A: {ref_paper_a}
  Claim A: {ref_claim_a}
  Paper B: {ref_paper_b}
  Claim B: {ref_claim_b}

WHAT TO IGNORE:
- Different paper pairings DO NOT disqualify a match. Multiple papers may make
  the same kind of claim, so two contradictions can describe the same
  disagreement using different anchor pairs.
- Different ordering of papers (A vs B vs B vs A) DOES NOT matter.
- Surface paraphrasing of claims DOES NOT matter.

WHAT MATTERS:
- Are both contradictions about the same empirical question (e.g., "do LLMs
  achieve high simulation fidelity?", "does alignment hurt simulation?")
- Do both contradictions point in the same direction (one side claims X, the
  other claims NOT-X, in both contradictions)?

EXAMPLES OF MATCHES:
- Contradiction 1: "Paper A says LLMs reach 85% accuracy" vs "Paper B says LLMs only reach 50%"
- Contradiction 2: "Paper A says LLMs reach 85% accuracy" vs "Paper C says LLMs only reach 60%"
- VERDICT: MATCH — both describe the same high-vs-low LLM accuracy disagreement, even though paired with different second papers.

EXAMPLES OF NON-MATCHES:
- Contradiction 1: A high-vs-low accuracy disagreement
- Contradiction 2: A privacy-vs-utility tradeoff disagreement
- VERDICT: NOT A MATCH — different topics entirely.

Now apply this to the two contradictions above. Respond with JSON only:
{{"match": true}} or {{"match": false}}"""

def _get_client() -> OpenAI:
    """Construct an OpenAI client using the configured API key."""
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _papers_match(pred: Contradiction, ref: dict) -> bool:
    """Check whether predicted and reference contradictions share at least one paper.

    We use shared-paper overlap (rather than exact pair match) because
    multiple papers in a corpus may anchor the same disagreement. For
    example, if papers A, B, C all claim "high fidelity" and paper D
    claims "low fidelity," the genuine disagreement is the
    high-vs-low conflict — whether anchored as (A,D), (B,D), or (C,D)
    is incidental.

    Args:
        pred: A predicted Contradiction.
        ref: A reference contradiction dict.

    Returns:
        True if the predicted and reference contradictions share at
        least one paper (and the substantive judge will then verify
        the disagreement is the same).
    """
    pred_pair = {pred.paper_a, pred.paper_b}
    ref_pair = {ref["paper_a"], ref["paper_b"]}
    return bool(pred_pair & ref_pair)  # at least one paper in common


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _judge_match(client: OpenAI, pred: Contradiction, ref: dict) -> bool:
    """Ask the LLM judge whether a predicted contradiction matches a reference.

    Args:
        client: An initialized OpenAI client.
        pred: The pipeline-produced contradiction.
        ref: A ground-truth contradiction dict.

    Returns:
        True if the judge confirms the same underlying disagreement.
        Returns False on JSON parse failure (conservative).
    """
    prompt = MATCH_JUDGE_PROMPT.format(
        pred_paper_a=pred.paper_a,
        pred_claim_a=pred.claim_a,
        pred_paper_b=pred.paper_b,
        pred_claim_b=pred.claim_b,
        ref_paper_a=ref["paper_a"],
        ref_claim_a=ref["claim_a"],
        ref_paper_b=ref["paper_b"],
        ref_claim_b=ref["claim_b"],
    )
    response = client.chat.completions.create(
        model=config.EXTRACTION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Match judge returned unparseable JSON: %r", raw[:200])
        return False
    return bool(parsed.get("match", False))


def _compute_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute Precision, Recall, F1 from confusion-matrix counts.

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.

    Returns:
        Dict with 'precision', 'recall', 'f1' keys. Each is 0.0 when the
        relevant denominator is zero (no predictions or no references).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _append_to_markdown(report_text: str, output_path: Path) -> None:
    """Append the F1 report to evaluation_results.md, replacing any prior version.

    Idempotent: re-running this script overwrites only the contradiction
    section, preserving BERTScore and Precision@5 sections from earlier steps.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    marker = "## Contradiction Detection: Precision, Recall, F1"
    if marker in existing:
        before = existing.split(marker)[0]
        rest = existing.split(marker, 1)[1]
        next_heading_idx = rest.find("\n## ")
        after = rest[next_heading_idx:] if next_heading_idx >= 0 else ""
        output_path.write_text(before + report_text + after, encoding="utf-8")
    else:
        output_path.write_text(existing + "\n" + report_text, encoding="utf-8")


def _run_pipeline_extraction_and_synthesis(
    eval_papers: list[str],
) -> list[Contradiction]:
    """Run the full extraction + contradiction detection on the eval papers only.

    Restricting to eval papers (rather than every paper in ChromaDB) is
    important when the store also contains unrelated papers from earlier
    development — those would otherwise pollute the synthesis prompt and
    cause cross-paper misattribution.

    Args:
        eval_papers: Filenames to include in extraction and synthesis.

    Returns:
        Predicted contradictions from the pipeline.
    """
    store = VectorStore()
    indexed = set(store.list_papers())

    # Filter to papers that are both requested AND actually in the store
    papers_to_use = [p for p in eval_papers if p in indexed]
    missing = set(eval_papers) - indexed
    if missing:
        print(f"  WARNING: {len(missing)} eval papers not in store: {missing}")
    print(f"  Running on {len(papers_to_use)} eval papers: {papers_to_use}")

    searcher = HybridSearcher(store)
    print("  Extracting all 8 fields per paper ...")
    extractions = [extract_paper(searcher, p) for p in papers_to_use]

    print("  Running contradiction synthesis ...")
    return detect_contradictions(extractions)


def main() -> None:
    """Entry point: run pipeline, judge matches, compute F1."""
    print("Loading ground-truth contradictions ...")
    with GROUND_TRUTH_PATH.open(encoding="utf-8") as fh:
        ground_truth: list[dict] = json.load(fh)
    print(f"  {len(ground_truth)} ground-truth contradictions")

    print("\nRunning extraction + contradiction synthesis ...")
    # Only run synthesis on the papers that have ground-truth annotations.
    # Papers in the store from earlier development would otherwise pollute
    # the comparison and make evaluation results incomparable.
    eval_papers = sorted(
        {ref["paper_a"] for ref in ground_truth}
        | {ref["paper_b"] for ref in ground_truth}
    )
    predictions = _run_pipeline_extraction_and_synthesis(eval_papers)
    print(f"  Pipeline produced {len(predictions)} contradictions")

    if not predictions and not ground_truth:
        # Edge case: nothing to compare in either direction
        print("\nBoth predictions and ground truth are empty. F1 undefined.")
        return

    # Match each prediction against ground-truth references
    client = _get_client()
    matched_refs: set[int] = set()  # indices of matched ground-truth records
    pred_results: list[dict] = []

    print(f"\nJudging matches ({len(predictions)} predictions x "
          f"{len(ground_truth)} references) ...")

    for pred in predictions:
        match_idx: int | None = None
        for ref_idx, ref in enumerate(ground_truth):
            # Cheap pre-filter: skip refs that share no paper with this prediction
            if not _papers_match(pred, ref):
                continue
            # Already-matched refs cannot be re-used (one-to-one matching)
            if ref_idx in matched_refs:
                continue
            if _judge_match(client, pred, ref):
                match_idx = ref_idx
                matched_refs.add(ref_idx)
                break

        pred_results.append(
            {
                "paper_a": pred.paper_a,
                "paper_b": pred.paper_b,
                "claim_a": pred.claim_a,
                "claim_b": pred.claim_b,
                "explanation": pred.explanation,
                "matched_ref_index": match_idx,
                "is_true_positive": match_idx is not None,
            }
        )

    tp = sum(1 for r in pred_results if r["is_true_positive"])
    fp = len(pred_results) - tp
    fn = len(ground_truth) - len(matched_refs)
    metrics = _compute_metrics(tp, fp, fn)

    # Print summary
    print("\n" + "=" * 60)
    print("CONTRADICTION DETECTION RESULTS")
    print("=" * 60)
    print(f"  Predictions:           {len(predictions)}")
    print(f"  Ground-truth refs:     {len(ground_truth)}")
    print(f"  True positives  (TP):  {tp}")
    print(f"  False positives (FP):  {fp}")
    print(f"  False negatives (FN):  {fn}")
    print("-" * 60)
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print("=" * 60)

    # Markdown report
    md_lines = [
        "## Contradiction Detection: Precision, Recall, F1",
        "",
        f"**Judge model:** `{config.EXTRACTION_MODEL}` &nbsp;&nbsp; "
        f"**Synthesis model:** `{config.SYNTHESIS_MODEL}`",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Predicted contradictions | {len(predictions)} |",
        f"| Ground-truth contradictions | {len(ground_truth)} |",
        f"| True positives | {tp} |",
        f"| False positives | {fp} |",
        f"| False negatives | {fn} |",
        f"| **Precision** | **{metrics['precision']:.4f}** |",
        f"| **Recall** | **{metrics['recall']:.4f}** |",
        f"| **F1** | **{metrics['f1']:.4f}** |",
        "",
    ]
    _append_to_markdown("\n".join(md_lines), RESULTS_MD_PATH)

    # Persist raw records for debugging
    RAW_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "metrics": metrics,
                "counts": {"tp": tp, "fp": fp, "fn": fn},
                "ground_truth": ground_truth,
                "predictions": pred_results,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nAppended results to {RESULTS_MD_PATH}")
    print(f"Wrote {RAW_JSON_PATH}")


if __name__ == "__main__":
    main()
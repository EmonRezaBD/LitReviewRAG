"""BERTScore evaluation of field extraction quality.

Compares pipeline-extracted field values against human-annotated ground
truth using BERTScore — a contextual-embedding similarity metric that
correlates well with human judgment for paraphrased or near-equivalent
text. Reports per-field and overall Precision, Recall, and F1.

Output:
- Aggregate metrics printed to stdout.
- Per-field detail saved to docs/evaluation_results.md (markdown table).
- Raw per-pair scores saved to results/bertscore_raw.json for debugging.

Usage:
    python -m evaluation.bertscore_eval
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from bert_score import score as bert_score
from openpyxl import load_workbook

from litreviewrag.extraction.extractor import extract_paper
from litreviewrag.extraction.prompts import FIELD_SPECS
from litreviewrag.retrieval.hybrid_search import HybridSearcher
from litreviewrag.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = Path("evaluation/ground_truth/gold_standard.xlsx")
RESULTS_MD_PATH = Path("docs/evaluation_results.md")
RAW_JSON_PATH = Path("results/bertscore_raw.json")

# BERTScore base model. roberta-large is the standard choice and what
# the original BERTScore paper uses for English text.
BERT_MODEL = "roberta-large"


@dataclass
class FieldScore:
    """BERTScore P/R/F1 for one (paper, field) pair.

    Attributes:
        paper_name: Source paper filename.
        field_name: Which field was scored.
        precision: BERTScore Precision (token-level alignment from prediction to reference).
        recall: BERTScore Recall (token-level alignment from reference to prediction).
        f1: Harmonic mean of P and R.
        prediction: The pipeline's extracted value (what was scored).
        reference: The ground-truth value (what it was compared against).
    """

    paper_name: str
    field_name: str
    precision: float
    recall: float
    f1: float
    prediction: str
    reference: str


def _load_ground_truth(path: Path) -> dict[str, dict[str, str]]:
    """Load gold-standard annotations from the Excel file.

    Args:
        path: Path to gold_standard.xlsx.

    Returns:
        Nested dict: {paper_name: {field_name: value_string}}. Empty cells
        in the Excel are normalized to empty strings, which BERTScore
        handles as a degenerate-but-valid input.
    """
    wb = load_workbook(path, read_only=True)
    ws = wb.active
    headers = [c.value for c in ws[1]]

    data: dict[str, dict[str, str]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        record = dict(zip(headers, row))
        paper = record.get("paper_name")
        if not paper:
            continue
        # Normalize None/missing -> empty string for fair comparison
        data[paper] = {
            spec.name: (record.get(spec.name) or "").strip()
            for spec in FIELD_SPECS
        }
    return data


def _run_extractions(paper_names: list[str]) -> dict[str, dict[str, str]]:
    """Run the pipeline on each paper and collect predicted field values.

    Args:
        paper_names: Filenames to extract (must already be in ChromaDB).

    Returns:
        Nested dict: {paper_name: {field_name: predicted_value_string}}.
    """
    store = VectorStore()
    searcher = HybridSearcher(store)
    indexed = set(store.list_papers())

    predictions: dict[str, dict[str, str]] = {}
    for paper in paper_names:
        if paper not in indexed:
            logger.warning(
                "Paper '%s' not found in ChromaDB. Skipping. "
                "Run `python -m litreviewrag ingest` first.",
                paper,
            )
            continue
        print(f"  Extracting {paper} ...")
        ex = extract_paper(searcher, paper)
        predictions[paper] = {
            name: (fe.value or "") for name, fe in ex.fields.items()
        }
    return predictions


def _compute_bertscores(
    predictions: list[str],
    references: list[str],
) -> tuple[list[float], list[float], list[float]]:
    """Run BERTScore over parallel prediction/reference lists.

    Args:
        predictions: List of pipeline outputs.
        references: List of ground-truth values (same length).

    Returns:
        Tuple (P_list, R_list, F1_list), each a list of floats per pair.
    """
    # bert_score returns torch tensors; convert to plain Python floats
    P, R, F1 = bert_score(
        cands=predictions,
        refs=references,
        model_type=BERT_MODEL,
        lang="en",
        verbose=False,
        rescale_with_baseline=False,  # raw scores are easier to interpret
    )
    return P.tolist(), R.tolist(), F1.tolist()


def _aggregate_by_field(scores: list[FieldScore]) -> dict[str, dict[str, float]]:
    """Compute mean P/R/F1 per field across all papers.

    Args:
        scores: All FieldScore records.

    Returns:
        {field_name: {'precision': mean, 'recall': mean, 'f1': mean, 'n': count}}
    """
    by_field: dict[str, list[FieldScore]] = {}
    for s in scores:
        by_field.setdefault(s.field_name, []).append(s)

    aggregated: dict[str, dict[str, float]] = {}
    for field_name, items in by_field.items():
        n = len(items)
        aggregated[field_name] = {
            "precision": sum(i.precision for i in items) / n,
            "recall": sum(i.recall for i in items) / n,
            "f1": sum(i.f1 for i in items) / n,
            "n": n,
        }
    return aggregated


def _write_markdown_report(
    overall: dict[str, float],
    by_field: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Write a human-readable markdown report of the evaluation.

    Args:
        overall: Aggregate {'precision', 'recall', 'f1', 'n'} across all pairs.
        by_field: Per-field aggregates from _aggregate_by_field.
        output_path: Where to write the .md file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation Results",
        "",
        "## BERTScore: Field Extraction Quality",
        "",
        f"**Model:** `{BERT_MODEL}` &nbsp;&nbsp; **Pairs evaluated:** {overall['n']}",
        "",
        "### Overall",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Precision | {overall['precision']:.4f} |",
        f"| Recall    | {overall['recall']:.4f} |",
        f"| **F1**    | **{overall['f1']:.4f}** |",
        "",
        "### Per-Field Breakdown",
        "",
        "| Field | Precision | Recall | F1 | n |",
        "|---|---|---|---|---|",
    ]
    # Preserve canonical field ordering from FIELD_SPECS
    for spec in FIELD_SPECS:
        m = by_field.get(spec.name)
        if not m:
            continue
        lines.append(
            f"| {spec.name} | {m['precision']:.4f} | {m['recall']:.4f} "
            f"| {m['f1']:.4f} | {int(m['n'])} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Entry point: load ground truth, run extractions, compute BERTScore."""
    print("Loading ground truth ...")
    ground_truth = _load_ground_truth(GROUND_TRUTH_PATH)
    paper_names = list(ground_truth.keys())
    print(f"  Loaded {len(paper_names)} papers, "
          f"{sum(len(f) for f in ground_truth.values())} field values")

    print("\nRunning pipeline extractions ...")
    predictions = _run_extractions(paper_names)

    # Build parallel lists for BERTScore. Skip pairs where the paper
    # was not indexed (logged warning above).
    pred_strings: list[str] = []
    ref_strings: list[str] = []
    pair_meta: list[tuple[str, str]] = []  # (paper, field)
    for paper in paper_names:
        if paper not in predictions:
            continue
        for spec in FIELD_SPECS:
            pred_strings.append(predictions[paper].get(spec.name, ""))
            ref_strings.append(ground_truth[paper].get(spec.name, ""))
            pair_meta.append((paper, spec.name))

    if not pred_strings:
        print("ERROR: No prediction/reference pairs to score. Aborting.")
        return

    print(f"\nComputing BERTScore on {len(pred_strings)} pairs "
          f"(downloading {BERT_MODEL} on first run, ~440MB) ...")
    P, R, F1 = _compute_bertscores(pred_strings, ref_strings)

    # Assemble per-pair records
    field_scores = [
        FieldScore(
            paper_name=paper,
            field_name=field,
            precision=P[i],
            recall=R[i],
            f1=F1[i],
            prediction=pred_strings[i],
            reference=ref_strings[i],
        )
        for i, (paper, field) in enumerate(pair_meta)
    ]

    # Aggregate
    overall = {
        "precision": sum(P) / len(P),
        "recall": sum(R) / len(R),
        "f1": sum(F1) / len(F1),
        "n": len(P),
    }
    by_field = _aggregate_by_field(field_scores)

    # Print summary
    print("\n" + "=" * 60)
    print(f"OVERALL  P={overall['precision']:.4f}  "
          f"R={overall['recall']:.4f}  F1={overall['f1']:.4f}  "
          f"(n={overall['n']})")
    print("=" * 60)
    print(f"{'Field':<22} {'P':>8} {'R':>8} {'F1':>8}  n")
    for spec in FIELD_SPECS:
        m = by_field.get(spec.name)
        if not m:
            continue
        print(f"{spec.name:<22} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f}  {int(m['n'])}")
    print("=" * 60)

    # Persist outputs
    _write_markdown_report(overall, by_field, RESULTS_MD_PATH)
    RAW_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(
            [
                {
                    "paper_name": s.paper_name,
                    "field_name": s.field_name,
                    "precision": s.precision,
                    "recall": s.recall,
                    "f1": s.f1,
                    "prediction": s.prediction,
                    "reference": s.reference,
                }
                for s in field_scores
            ],
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nWrote {RESULTS_MD_PATH}")
    print(f"Wrote {RAW_JSON_PATH}")


if __name__ == "__main__":
    main()
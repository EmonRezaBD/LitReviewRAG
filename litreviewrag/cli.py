"""Command-line interface for LitReviewRAG.

Exposes five subcommands that wrap the pipeline modules:

    ingest      Parse, chunk, embed, and store PDFs in ChromaDB.
    extract     Run 8-field extraction on indexed papers.
    contradict  Detect cross-paper contradictions.
    export      Write the final Excel deliverable.
    run-all     Execute the full pipeline end-to-end.
    demo        Quick demo on data/sample_papers/ → results/demo.xlsx.

Each subcommand is callable both standalone and via `run-all`. This
keeps individual stages debuggable while still offering a one-command
experience for graders and end users.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import typer

from litreviewrag import config
from litreviewrag.export.excel_writer import export_to_excel
from litreviewrag.extraction.extractor import (
    PaperExtraction,
    extract_paper,
)
from litreviewrag.extraction.prompts import FIELD_SPECS
from litreviewrag.ingestion.chunker import chunk_text
from litreviewrag.ingestion.pdf_parser import parse_pdf
from litreviewrag.retrieval.embeddings import embed_texts
from litreviewrag.retrieval.hybrid_search import HybridResult, HybridSearcher
from litreviewrag.retrieval.vector_store import VectorStore
from litreviewrag.synthesis.contradiction import (
    Contradiction,
    detect_contradictions,
)

# typer auto-generates --help text from these strings; keep them concise.
app = typer.Typer(
    name="litreviewrag",
    help="Automated literature review extraction and synthesis.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialization helpers — used so `extract` can save results to disk
# and `contradict`/`export` can read them back without re-running expensive
# LLM calls.
# ---------------------------------------------------------------------------


def _extraction_to_dict(ex: PaperExtraction) -> dict:
    """Serialize a PaperExtraction to a JSON-safe dict.

    Source chunks are flattened to (paper, chunk_index, text) triples so
    the output can round-trip through disk without losing citation info.

    Args:
        ex: The extraction result to serialize.

    Returns:
        A nested dict suitable for json.dump.
    """
    return {
        "paper_name": ex.paper_name,
        "fields": {
            name: {
                "value": fe.value,
                "source_chunks": [
                    {
                        "paper_name": c.paper_name,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                        "score": c.score,
                    }
                    for c in fe.source_chunks
                ],
            }
            for name, fe in ex.fields.items()
        },
    }


def _dict_to_extraction(d: dict) -> PaperExtraction:
    """Deserialize a dict back into a PaperExtraction.

    Args:
        d: Output of _extraction_to_dict.

    Returns:
        Reconstructed PaperExtraction with HybridResult source chunks.
    """
    # Local imports avoid circulars at module-load time
    from litreviewrag.extraction.extractor import FieldExtraction

    fields = {}
    for name, payload in d["fields"].items():
        chunks = [
            HybridResult(
                text=c["text"],
                paper_name=c["paper_name"],
                chunk_index=c["chunk_index"],
                # Round-tripped scores are informational only at this point
                score=c.get("score", 0.0),
                bm25_score=0.0,
                vector_score=0.0,
            )
            for c in payload.get("source_chunks", [])
        ]
        fields[name] = FieldExtraction(
            field_name=name,
            value=payload.get("value"),
            source_chunks=chunks,
        )
    return PaperExtraction(paper_name=d["paper_name"], fields=fields)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    input_dir: Path = typer.Option(
        Path("data/sample_papers"),
        "--input",
        "-i",
        help="Directory containing PDFs to ingest.",
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Wipe the ChromaDB store before ingesting (use when replacing PDFs with the same filename).",
    ),
) -> None:
    """Parse, chunk, embed, and store every PDF in INPUT_DIR."""
    config.validate()

    # Optional clean slate. Useful when the user replaces a paper file
    # without changing its filename — chunks would otherwise mix old + new.
    if reset and config.CHROMA_PERSIST_DIR.exists():
        typer.echo(f"Resetting {config.CHROMA_PERSIST_DIR} ...")
        shutil.rmtree(config.CHROMA_PERSIST_DIR)

    store = VectorStore()
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        typer.echo(f"No PDFs found in {input_dir}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Ingesting {len(pdfs)} PDFs from {input_dir} ...")
    for pdf_path in pdfs:
        typer.echo(f"  - {pdf_path.name}")
        text = parse_pdf(pdf_path)
        chunks = chunk_text(text)
        vectors = embed_texts([c.text for c in chunks])
        store.add_chunks(chunks, vectors, paper_name=pdf_path.name)

    typer.echo(f"Done. Total chunks: {store.count()}")
    typer.echo(f"Papers: {store.list_papers()}")


@app.command()
def extract(
    output: Path = typer.Option(
        Path("results/extractions.json"),
        "--output",
        "-o",
        help="Where to write the JSON extraction results.",
    ),
    top_k: int = typer.Option(
        5, "--top-k", help="Chunks to retrieve per field (default 5)."
    ),
    model: str = typer.Option(
        None,
        "--model",
        help=f"LLM to use (default: {config.EXTRACTION_MODEL}).",
    ),
) -> None:
    """Extract all 8 fields from every indexed paper."""
    config.validate()

    store = VectorStore()
    searcher = HybridSearcher(store)
    papers = store.list_papers()
    if not papers:
        typer.echo("No papers in store. Run `ingest` first.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Extracting {len(papers)} papers using {model or config.EXTRACTION_MODEL} ...")
    extractions = [
        extract_paper(searcher, p, model=model, top_k=top_k) for p in papers
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(
            [_extraction_to_dict(ex) for ex in extractions],
            fh,
            indent=2,
            ensure_ascii=False,
        )
    typer.echo(f"Wrote extractions for {len(extractions)} papers to {output}")


@app.command()
def contradict(
    extractions_path: Path = typer.Option(
        Path("results/extractions.json"),
        "--input",
        "-i",
        help="Path to the JSON file produced by `extract`.",
    ),
    output: Path = typer.Option(
        Path("results/contradictions.json"),
        "--output",
        "-o",
        help="Where to write the contradictions JSON.",
    ),
    model: str = typer.Option(
        None,
        "--model",
        help=f"Synthesis LLM (default: {config.SYNTHESIS_MODEL}).",
    ),
) -> None:
    """Detect cross-paper contradictions over extracted findings."""
    config.validate()

    if not extractions_path.exists():
        typer.echo(f"Not found: {extractions_path}. Run `extract` first.", err=True)
        raise typer.Exit(code=1)

    with extractions_path.open(encoding="utf-8") as fh:
        records = json.load(fh)
    extractions = [_dict_to_extraction(r) for r in records]

    typer.echo(f"Detecting contradictions across {len(extractions)} papers ...")
    contras = detect_contradictions(extractions, model=model)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(
            [c.__dict__ for c in contras],
            fh,
            indent=2,
            ensure_ascii=False,
        )
    typer.echo(f"Found {len(contras)} contradictions; wrote {output}")


@app.command()
def export(
    extractions_path: Path = typer.Option(
        Path("results/extractions.json"),
        "--extractions",
        help="JSON file from `extract`.",
    ),
    contradictions_path: Path = typer.Option(
        Path("results/contradictions.json"),
        "--contradictions",
        help="JSON file from `contradict`.",
    ),
    output: Path = typer.Option(
        Path("results/literature_review.xlsx"),
        "--output",
        "-o",
        help="Destination .xlsx file.",
    ),
) -> None:
    """Combine extractions + contradictions into a styled Excel file."""
    if not extractions_path.exists():
        typer.echo(f"Not found: {extractions_path}", err=True)
        raise typer.Exit(code=1)

    with extractions_path.open(encoding="utf-8") as fh:
        extractions = [_dict_to_extraction(r) for r in json.load(fh)]

    contras: list[Contradiction] = []
    if contradictions_path.exists():
        with contradictions_path.open(encoding="utf-8") as fh:
            contras = [Contradiction(**r) for r in json.load(fh)]
    else:
        typer.echo(f"Note: {contradictions_path} not found; exporting without contradictions.")

    export_to_excel(extractions, contras, output)
    typer.echo(f"Wrote {output}")


@app.command(name="run-all")
def run_all(
    input_dir: Path = typer.Option(
        Path("data/sample_papers"),
        "--input",
        "-i",
        help="Directory of PDFs to process.",
    ),
    output_dir: Path = typer.Option(
        Path("results"),
        "--output",
        "-o",
        help="Directory for all intermediate + final artifacts.",
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Wipe ChromaDB before ingesting.",
    ),
) -> None:
    """Run the full pipeline: ingest → extract → contradict → export."""
    config.validate()

    # 1. Ingest
    if reset and config.CHROMA_PERSIST_DIR.exists():
        shutil.rmtree(config.CHROMA_PERSIST_DIR)
    store = VectorStore()
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        typer.echo(f"No PDFs found in {input_dir}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[1/4] Ingesting {len(pdfs)} PDFs ...")
    for pdf_path in pdfs:
        text = parse_pdf(pdf_path)
        chunks = chunk_text(text)
        vectors = embed_texts([c.text for c in chunks])
        store.add_chunks(chunks, vectors, paper_name=pdf_path.name)

    # 2. Extract
    searcher = HybridSearcher(store)
    papers = store.list_papers()
    typer.echo(f"[2/4] Extracting {len(papers)} papers ...")
    extractions = [extract_paper(searcher, p) for p in papers]

    # 3. Contradict
    typer.echo(f"[3/4] Detecting contradictions ...")
    contras = detect_contradictions(extractions)
    typer.echo(f"      Found {len(contras)} contradictions")

    # 4. Export
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / "literature_review.xlsx"
    typer.echo(f"[4/4] Writing {excel_path} ...")
    export_to_excel(extractions, contras, excel_path)

    typer.echo(f"\n✅ Done. Open {excel_path} to view your literature review.")


@app.command()
def demo() -> None:
    """Run the full pipeline on data/sample_papers/ with a fresh store."""
    typer.echo("Running LitReviewRAG demo ...\n")
    run_all(
        input_dir=Path("data/sample_papers"),
        output_dir=Path("results"),
        reset=True,
    )


if __name__ == "__main__":
    app()
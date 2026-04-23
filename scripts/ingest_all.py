"""Ingest all PDFs in data/sample_papers/ into the ChromaDB vector store.

Run: python -m scripts.ingest_all
"""

import logging
from pathlib import Path

from litreviewrag.ingestion.chunker import chunk_text
from litreviewrag.ingestion.pdf_parser import parse_pdf
from litreviewrag.retrieval.embeddings import embed_texts
from litreviewrag.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PAPERS_DIR = Path("data/sample_papers")


def main() -> None:
    """Parse, chunk, embed, and upsert every PDF in PAPERS_DIR."""
    store = VectorStore()
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    logger.info("Found %d PDFs", len(pdfs))

    for pdf_path in pdfs:
        logger.info("Ingesting %s", pdf_path.name)
        text = parse_pdf(pdf_path)
        chunks = chunk_text(text)
        vectors = embed_texts([c.text for c in chunks])
        store.add_chunks(chunks, vectors, paper_name=pdf_path.name)
        logger.info("  -> %d chunks added", len(chunks))

    logger.info("Done. Total chunks in store: %d", store.count())
    logger.info("Papers: %s", store.list_papers())


if __name__ == "__main__":
    main()
"""ChromaDB wrapper for persistent vector storage and similarity search.

Stores chunks with their embeddings, source metadata (paper name, chunk
index, character offsets), and supports cosine-similarity queries. Uses
a local persistent client — no server or cloud account required.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from litreviewrag import config
from litreviewrag.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Default collection name for all ingested papers. Kept as a single
# collection so hybrid search can combine scores across the full corpus.
_COLLECTION_NAME = "litreview_chunks"


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from a similarity search, with distance score.

    Attributes:
        text: The chunk's text content.
        paper_name: Source PDF filename (e.g., 'sample1.pdf').
        chunk_index: Chunk's position within its source document.
        distance: Cosine distance from the query vector (lower = more similar).
    """

    text: str
    paper_name: str
    chunk_index: int
    distance: float


class VectorStore:
    """Persistent ChromaDB-backed store for chunk embeddings."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        """Initialize the store, creating the collection if it doesn't exist.

        Args:
            persist_dir: Directory where ChromaDB persists its files. Defaults
                to config.CHROMA_PERSIST_DIR.
        """
        self._persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # anonymized_telemetry=False prevents ChromaDB from sending usage pings
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # get_or_create avoids errors on repeated ingestions of the same corpus
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine is standard for embeddings
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        paper_name: str,
    ) -> None:
        """Add a paper's chunks and their embeddings to the store.

        Each chunk receives a globally-unique ID of the form
        "{paper_name}::chunk_{index}" so chunks from different papers
        coexist in the same collection without collision.

        Args:
            chunks: List of Chunk objects (must be same length as embeddings).
            embeddings: List of embedding vectors, one per chunk.
            paper_name: Source PDF filename, stored as metadata for filtering
                and citation.

        Raises:
            ValueError: If chunks and embeddings have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )
        if not chunks:
            return

        # Build parallel lists that ChromaDB expects
        ids = [f"{paper_name}::chunk_{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "paper_name": paper_name,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in chunks
        ]

        # upsert (not add) so re-ingesting the same paper overwrites cleanly
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks for paper %s", len(chunks), paper_name)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        paper_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """Find the top-k most similar chunks to a query embedding.

        Args:
            query_embedding: The query vector (must match collection dim).
            top_k: Number of results to return.
            paper_name: If provided, restricts search to chunks from this
                paper. Required for per-paper field extraction so chunks from
                other papers don't contaminate results.

        Returns:
            List of RetrievedChunk objects sorted by similarity (best first).
        """
        where_clause = {"paper_name": paper_name} if paper_name else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
        )

        # ChromaDB returns parallel lists wrapped in an outer list per query.
        # We only issue one query at a time, so index [0] everywhere.
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            RetrievedChunk(
                text=doc,
                paper_name=str(meta["paper_name"]),
                chunk_index=int(meta["chunk_index"]),
                distance=float(dist),
            )
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def list_papers(self) -> list[str]:
        """Return all unique paper_name values currently in the store.

        Returns:
            Sorted list of paper filenames present in the collection.
        """
        # get() with no filter pulls all metadata (cheap — no embeddings)
        all_records = self._collection.get(include=["metadatas"])
        paper_names = {str(m["paper_name"]) for m in all_records["metadatas"]}
        return sorted(paper_names)

    def count(self) -> int:
        """Return the total number of chunks in the store.

        Returns:
            Integer count of all chunks across all papers.
        """
        return self._collection.count()
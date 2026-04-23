"""Hybrid retrieval combining BM25 lexical search with dense vector search.

Dense embeddings excel at semantic similarity but can miss rare terms and
exact matches (acronyms, metric names, author names). BM25 excels at
lexical matching but misses paraphrases. Fusing both via weighted score
combination improves retrieval precision over either method alone.
"""

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from litreviewrag.retrieval.embeddings import embed_texts
from litreviewrag.retrieval.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

# Weight for the dense-vector score in the final fusion. BM25 gets (1 - alpha).
# 0.5 is a balanced default; tune based on evaluation.
_DEFAULT_ALPHA = 0.5


@dataclass(frozen=True)
class HybridResult:
    """A chunk ranked by fused BM25 + vector similarity.

    Attributes:
        text: The chunk's text content.
        paper_name: Source PDF filename.
        chunk_index: Position within source document.
        score: Fused relevance score (higher = more relevant).
        bm25_score: Normalized BM25 component of the fused score.
        vector_score: Normalized vector-similarity component.
    """

    text: str
    paper_name: str
    chunk_index: int
    score: float
    bm25_score: float
    vector_score: float


def _tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer for BM25.

    Splits on non-alphanumeric characters and lowercases the result. This
    is intentionally simple — BM25 is robust to tokenization variations.

    Args:
        text: Text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] via min-max scaling.

    Needed because BM25 scores and vector distances live on different
    scales. Normalizing both to [0, 1] makes weighted fusion meaningful.

    Args:
        scores: Raw scores to normalize.

    Returns:
        List of scores in [0, 1]. If all scores are equal, returns zeros.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        # All scores identical — can't meaningfully rank, return neutral 0.0
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


class HybridSearcher:
    """Combines BM25 and vector similarity for paper-scoped retrieval."""

    def __init__(self, store: VectorStore, alpha: float = _DEFAULT_ALPHA) -> None:
        """Initialize the hybrid searcher.

        Args:
            store: A populated VectorStore to query.
            alpha: Weight for the vector score in [0, 1]. BM25 gets (1 - alpha).
                Default 0.5 gives equal weight to both signals.

        Raises:
            ValueError: If alpha is outside [0, 1].
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._store = store
        self._alpha = alpha

    def search(
        self,
        query: str,
        paper_name: str,
        top_k: int = 5,
        candidate_pool: int = 20,
    ) -> list[HybridResult]:
        """Retrieve top-k chunks for a query using BM25 + vector fusion.

        The fusion strategy:
        1. Pull a larger candidate pool (default 20) from the vector store,
           restricted to chunks from `paper_name`.
        2. Score the same candidates with BM25 using tokenized query terms.
        3. Min-max normalize both score lists to [0, 1].
        4. Fuse: final_score = alpha * vector + (1 - alpha) * bm25
        5. Return the top-k by fused score.

        Args:
            query: The natural-language query (e.g., "What is the main
                problem this paper addresses?").
            paper_name: Restrict search to this paper's chunks.
            top_k: Number of final results to return.
            candidate_pool: Initial vector-search pool size before BM25
                re-ranking. Larger = more thorough but slower.

        Returns:
            Top-k HybridResult objects sorted by fused score (best first).
        """
        # Step 1: embed the query and pull a candidate pool from vector search.
        # We retrieve more candidates than top_k to give BM25 room to re-rank.
        query_vec = embed_texts([query])[0]
        candidates: list[RetrievedChunk] = self._store.query(
            query_embedding=query_vec,
            top_k=candidate_pool,
            paper_name=paper_name,
        )
        if not candidates:
            return []

        # Step 2: score all candidates with BM25 using the query terms.
        # BM25 is fit on the candidate pool itself, which is fine for
        # per-query ranking (we don't need global IDF).
        tokenized_corpus = [_tokenize(c.text) for c in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = _tokenize(query)
        bm25_raw_scores = list(bm25.get_scores(tokenized_query))

        # Step 3: convert vector distances to similarities, then normalize.
        # ChromaDB returns cosine *distance* (lower is better); invert to
        # similarity (higher is better) so both signals point the same way.
        vector_similarities = [1.0 - c.distance for c in candidates]
        vec_norm = _min_max_normalize(vector_similarities)
        bm25_norm = _min_max_normalize(bm25_raw_scores)

        # Step 4: weighted fusion.
        fused: list[HybridResult] = []
        for cand, v_score, b_score in zip(candidates, vec_norm, bm25_norm):
            final = self._alpha * v_score + (1.0 - self._alpha) * b_score
            fused.append(
                HybridResult(
                    text=cand.text,
                    paper_name=cand.paper_name,
                    chunk_index=cand.chunk_index,
                    score=final,
                    bm25_score=b_score,
                    vector_score=v_score,
                )
            )

        # Step 5: return top-k by fused score.
        fused.sort(key=lambda r: r.score, reverse=True)
        return fused[:top_k]
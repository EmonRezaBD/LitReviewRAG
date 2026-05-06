"""Unit tests for hybrid-search helper utilities.

Covers the pure-logic helpers in hybrid_search.py without requiring
a live ChromaDB instance: tokenization, score normalization, and
alpha parameter validation.
"""

import pytest

from litreviewrag.retrieval.hybrid_search import (
    HybridSearcher,
    _min_max_normalize,
    _tokenize,
)


def test_tokenizer_lowercases_and_splits_on_non_alnum():
    """Tokenizer should produce lowercase alphanumeric tokens only."""
    tokens = _tokenize("BERTScore F1: 0.85! Hello-World")
    assert "bertscore" in tokens
    assert "f1" in tokens
    assert "0" in tokens or "85" in tokens
    assert "hello" in tokens
    assert "world" in tokens
    # No punctuation, no uppercase
    assert all(t.islower() or t.isdigit() for t in tokens)
    assert "BERTScore" not in tokens


def test_tokenizer_handles_empty_string():
    """Empty input should produce no tokens, not an error."""
    assert _tokenize("") == []


def test_min_max_normalize_maps_to_zero_one_range():
    """All output values must lie in [0, 1] with min=0 and max=1."""
    normalized = _min_max_normalize([3.0, 1.0, 5.0, 2.0, 4.0])
    assert min(normalized) == 0.0
    assert max(normalized) == 1.0
    assert all(0.0 <= v <= 1.0 for v in normalized)


def test_min_max_normalize_handles_constant_input():
    """If all values are equal, normalization returns zeros (no signal).

    This avoids divide-by-zero on degenerate inputs (e.g., all chunks
    receiving identical BM25 scores).
    """
    assert _min_max_normalize([2.5, 2.5, 2.5]) == [0.0, 0.0, 0.0]


def test_min_max_normalize_empty_input():
    """Empty input returns empty output, not an error."""
    assert _min_max_normalize([]) == []


def test_alpha_must_be_in_zero_one_range():
    """HybridSearcher rejects alpha outside [0, 1].

    We pass None for the store because the constructor doesn't touch it
    until a query is issued; alpha validation happens first.
    """
    with pytest.raises(ValueError):
        HybridSearcher(store=None, alpha=-0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HybridSearcher(store=None, alpha=1.5)  # type: ignore[arg-type]
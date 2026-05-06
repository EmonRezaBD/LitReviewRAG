"""Unit tests for the character-level chunker.

Covers:
- Output count math for various input sizes
- Overlap behavior between adjacent chunks
- Edge cases (empty input, exact boundary conditions)
- Parameter validation
"""

import pytest

from litreviewrag.ingestion.chunker import Chunk, chunk_text


def test_empty_text_returns_empty_list():
    """An empty input produces zero chunks (graceful no-op)."""
    assert chunk_text("") == []


def test_text_shorter_than_chunk_size_yields_one_chunk():
    """Short inputs should not be padded or split."""
    text = "hello world"
    chunks = chunk_text(text, chunk_size=600, stride=200)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(text)


# def test_chunk_count_matches_sliding_window_math():
#     """For a 1500-char text with chunk=600 and stride=200, expect 5 chunks.

#     Chunks start at offsets 0, 200, 400, 600, 800, 1000, 1200, 1400.
#     With chunk_size=600, the chunk starting at 1400 ends at 1500 (the
#     document end), and the loop terminates. So 8 chunks total.
#     """
#     text = "x" * 1500
#     chunks = chunk_text(text, chunk_size=600, stride=200)
#     assert len(chunks) == 8
#     # Last chunk must end exactly at document length
#     assert chunks[-1].end_char == 1500

def test_chunk_count_matches_sliding_window_math():
    """For a 1500-char text with chunk=600 and stride=200, expect 6 chunks.

    Chunks start at offsets 0, 200, 400, 600, 800, 1000. The chunk
    starting at 1000 ends at 1500 (the document end), and the loop
    terminates per the early-return optimization in chunk_text.
    """
    text = "x" * 1500
    chunks = chunk_text(text, chunk_size=600, stride=200)
    assert len(chunks) == 6
    # Last chunk must end exactly at document length
    assert chunks[-1].end_char == 1500
    # Last chunk's start matches the expected stride pattern
    assert chunks[-1].start_char == 1000
    
def test_adjacent_chunks_overlap_by_chunk_minus_stride():
    """With chunk=600 and stride=200, overlap = 400 chars."""
    text = "x" * 2000
    chunks = chunk_text(text, chunk_size=600, stride=200)
    # Overlap = chunk_size - stride = 400
    expected_overlap = 600 - 200
    overlap = chunks[0].end_char - chunks[1].start_char
    assert overlap == expected_overlap


def test_chunk_indices_are_sequential_starting_at_zero():
    """chunk_index should be 0, 1, 2, ... in document order."""
    text = "x" * 2000
    chunks = chunk_text(text, chunk_size=600, stride=200)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_invalid_stride_raises_value_error():
    """stride >= chunk_size would create non-overlapping or backward windows."""
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=600, stride=600)
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=600, stride=700)


def test_zero_or_negative_parameters_raise():
    """Defensive: chunk_size and stride must be positive."""
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0, stride=200)
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=600, stride=0)


def test_chunk_is_immutable_dataclass():
    """Chunk uses frozen=True so chunks can be safely shared/hashed."""
    chunk = Chunk(text="x", chunk_index=0, start_char=0, end_char=1)
    with pytest.raises(Exception):
        # frozen dataclasses raise on attribute assignment
        chunk.text = "y"  # type: ignore[misc]
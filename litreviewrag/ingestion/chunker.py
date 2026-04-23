"""Character-level text chunker with overlapping windows.

Splits raw paper text into fixed-size chunks with a configurable stride.
Overlap between adjacent chunks preserves context across boundaries,
improving retrieval recall for queries whose answer spans a chunk edge.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A single text chunk with positional metadata.

    Attributes:
        text: The chunk's text content.
        chunk_index: Zero-based index of this chunk within its source document.
        start_char: Inclusive start offset in the source text.
        end_char: Exclusive end offset in the source text.
    """

    text: str
    chunk_index: int
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    chunk_size: int = 600,
    stride: int = 200,
) -> list[Chunk]:
    """Split text into overlapping character-level chunks.

    Uses a sliding window where each chunk starts `stride` characters after
    the previous chunk. With the default 600/200 settings, adjacent chunks
    overlap by 400 characters, preserving context across boundaries.

    Args:
        text: Raw text to chunk (typically PDF-extracted paper text).
        chunk_size: Length in characters per chunk. Default 600 matches the
            proposal specification.
        stride: Step size between chunk starts. Default 200 yields a
            400-character overlap. Must be strictly less than chunk_size.

    Returns:
        List of Chunk objects in document order. Empty list if input is empty.

    Raises:
        ValueError: If chunk_size <= 0 or stride <= 0 or stride >= chunk_size.
    """
    # Validate parameters before doing any work
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if stride >= chunk_size:
        raise ValueError(
            f"stride ({stride}) must be < chunk_size ({chunk_size}) "
            "to create overlapping windows"
        )

    if not text:
        return []

    chunks: list[Chunk] = []
    text_len = len(text)
    start = 0
    chunk_index = 0

    # Slide the window forward by `stride` until we cover the whole document
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text_content = text[start:end]
        chunks.append(
            Chunk(
                text=chunk_text_content,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
            )
        )
        chunk_index += 1

        # If this chunk reached the end of the text, we're done.
        # Prevents an extra tiny trailing chunk.
        if end == text_len:
            break

        start += stride

    return chunks
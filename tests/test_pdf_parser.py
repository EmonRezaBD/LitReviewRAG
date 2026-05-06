"""Unit tests for the PDF parser.

Covers:
- File-not-found error path
- Type acceptance (Path object vs string path)

Note: Live parsing of real PDFs is exercised by the smoke tests in
the development workflow; these unit tests focus on error handling
and input validation that don't require a real PDF on disk.
"""

from pathlib import Path

import pytest

from litreviewrag.ingestion.pdf_parser import parse_pdf


def test_missing_file_raises_file_not_found():
    """Calling parse_pdf on a nonexistent path should raise the standard error."""
    with pytest.raises(FileNotFoundError):
        parse_pdf(Path("/no/such/path/missing.pdf"))


def test_string_path_is_accepted():
    """parse_pdf must accept str inputs, not just Path objects.

    We expect a FileNotFoundError because the file doesn't exist, NOT
    a TypeError from rejecting the str.
    """
    with pytest.raises(FileNotFoundError):
        parse_pdf("nonexistent.pdf")  # type: ignore[arg-type]
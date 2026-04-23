"""PDF text extraction with fallback strategy.

Uses pdfplumber as the primary parser (better layout handling) and falls
back to pypdf if pdfplumber fails on a given file. Scanned PDFs with no
extractable text are reported and skipped by the caller.
"""

import logging
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFParseError(Exception):
    """Raised when neither pdfplumber nor pypdf can extract text."""


def parse_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file.

    Tries pdfplumber first (primary parser). If it fails or returns empty
    text, falls back to pypdf. Raises PDFParseError if both parsers fail
    or produce no extractable content (likely a scanned PDF).

    Args:
        pdf_path: Absolute or relative path to a .pdf file.

    Returns:
        Concatenated text from all pages, separated by double newlines.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        PDFParseError: If no parser can extract text from the file.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Try pdfplumber first — better at multi-column layouts and tables
    text = _parse_with_pdfplumber(pdf_path)
    if text.strip():
        return text

    # Fallback: pypdf is simpler but works on PDFs that break pdfplumber
    logger.warning("pdfplumber returned empty text for %s; trying pypdf", pdf_path.name)
    text = _parse_with_pypdf(pdf_path)
    if text.strip():
        return text

    # Both parsers failed — likely a scanned PDF needing OCR (out of scope)
    raise PDFParseError(
        f"No extractable text in {pdf_path.name}. "
        "File may be a scanned image requiring OCR."
    )


def _parse_with_pdfplumber(pdf_path: Path) -> str:
    """Extract text using pdfplumber. Returns empty string on failure."""
    try:
        pages: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
        return "\n\n".join(pages)
    except Exception as exc:  # broad catch: library raises many exception types
        logger.debug("pdfplumber failed on %s: %s", pdf_path.name, exc)
        return ""


def _parse_with_pypdf(pdf_path: Path) -> str:
    """Extract text using pypdf. Returns empty string on failure."""
    try:
        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except Exception as exc:
        logger.debug("pypdf failed on %s: %s", pdf_path.name, exc)
        return ""
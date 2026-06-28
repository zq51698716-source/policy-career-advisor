"""
PDF parser — extracts text and metadata from PDF files using PyMuPDF (fitz).

Replaces the deprecated PyPDF2 with PyMuPDF, which is faster, more accurate
for CJK text, and supports metadata extraction.

OCR fallback for scanned PDFs is detected automatically and reported to the
caller. Full OCR requires Tesseract to be installed separately.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Minimum character count to consider a PDF as "not scanned"
MIN_TEXT_LENGTH_FOR_NON_OCR = 100


def extract_text(file_path: str) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text as a single string (pages separated by newlines).

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If the file is not a valid PDF or is encrypted.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Cannot open PDF file '{file_path}': {e}") from e

    if doc.is_encrypted:
        doc.close()
        raise ValueError(f"PDF file is encrypted and cannot be read: {file_path}")

    text_parts = []
    for page_num, page in enumerate(doc, 1):
        try:
            t = page.get_text("text")  # text mode preserves reading order
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_num}: {e}")
            t = ""

        if t and t.strip():
            text_parts.append(t.strip())

    doc.close()
    return "\n\n".join(text_parts)


def extract_metadata(file_path: str) -> dict:
    """
    Extract PDF metadata: title, author, subject, creation date, page count, etc.

    Returns a dict compatible with the chunk metadata system.
    """
    try:
        doc = fitz.open(file_path)
        meta = doc.metadata or {}
        result = {
            "title": meta.get("title") or Path(file_path).stem,
            "author": meta.get("author") or "",
            "subject": meta.get("subject") or "",
            "creator": meta.get("creator") or "",
            "creation_date": _parse_pdf_date(meta.get("creationDate")),
            "modification_date": _parse_pdf_date(meta.get("modDate")),
            "page_count": doc.page_count,
            "file_size_bytes": os.path.getsize(file_path),
        }
        doc.close()
        return result
    except Exception as e:
        logger.warning(f"Failed to extract metadata from '{file_path}': {e}")
        return {
            "title": Path(file_path).stem,
            "author": "",
            "page_count": 0,
            "file_size_bytes": os.path.getsize(file_path),
        }


def extract_text_per_page(file_path: str) -> list[dict]:
    """
    Extract text page by page with page numbers.

    Useful for chunking strategies that need page-level granularity.

    Returns:
        [{"page_num": int, "text": str}, ...]
    """
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Cannot open PDF file '{file_path}': {e}") from e

    pages = []
    for page_num, page in enumerate(doc, 1):
        t = page.get_text("text")
        if t and t.strip():
            pages.append({
                "page_num": page_num,
                "text": t.strip(),
            })

    doc.close()
    return pages


def needs_ocr(file_path: str) -> bool:
    """
    Detect whether a PDF is likely a scanned document (needs OCR).

    Returns True if extracted text is very short, indicating a scanned image PDF.
    """
    text = extract_text(file_path)
    return len(text.strip()) < MIN_TEXT_LENGTH_FOR_NON_OCR


def extract_text_with_ocr(file_path: str) -> str:
    """
    Extract text from a scanned PDF using OCR.

    NOTE: This is a STUB — full OCR requires installing Tesseract + pytesseract.
    Currently returns text extracted by PyMuPDF and warns if OCR is needed.

    To enable OCR:
      1. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
      2. pip install pytesseract Pillow
      3. Replace this function body with actual OCR logic.
    """
    text = extract_text(file_path)

    if len(text.strip()) < MIN_TEXT_LENGTH_FOR_NON_OCR:
        logger.warning(
            f"PDF '{file_path}' appears to be a scanned document "
            f"(extracted only {len(text.strip())} chars). "
            f"OCR support requires Tesseract installation. "
            f"See backend/rag/pdf_parser.py:extract_text_with_ocr() for details."
        )

    return text


def _parse_pdf_date(date_str: Optional[str]) -> str:
    """Convert PDF date format (D:YYYYMMDDHHmmSS) to ISO format."""
    if not date_str:
        return ""
    # Strip "D:" prefix and timezone suffix
    cleaned = date_str.replace("D:", "").split("+")[0].split("-")[0].split("Z")[0]
    try:
        if len(cleaned) >= 14:
            dt = datetime.strptime(cleaned[:14], "%Y%m%d%H%M%S")
            return dt.isoformat()
        elif len(cleaned) >= 8:
            dt = datetime.strptime(cleaned[:8], "%Y%m%d")
            return dt.isoformat()
    except ValueError:
        pass
    return cleaned


# For backward compatibility, chunk_text is now in chunker.py
# This import is here so existing code doesn't break:
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Deprecated: use chunker.SemanticChunker instead.
    Kept for backward compatibility.
    """
    from backend.rag.chunker import chunk_text as _chunk_text
    return _chunk_text(text, chunk_size, overlap)

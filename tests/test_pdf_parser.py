"""
Unit tests for pdf_parser — PDF text extraction and metadata.

NOTE: These tests use PyMuPDF (fitz), which supersedes PyPDF2.
"""

import os
import pytest
from pathlib import Path

from backend.rag.pdf_parser import (
    extract_text,
    extract_metadata,
    needs_ocr,
    extract_text_per_page,
    extract_text_with_ocr,
    chunk_text,
)

# Path to test fixtures
TESTS_DIR = Path(__file__).parent
UPLOADS_DIR = Path(__file__).parent.parent / "uploads"


def _find_test_pdf() -> str:
    """Find a PDF file in the uploads directory for testing."""
    if UPLOADS_DIR.exists():
        for f in UPLOADS_DIR.iterdir():
            if f.suffix.lower() == ".pdf":
                return str(f)
    pytest.skip("No test PDF found in uploads/")


class TestExtractText:
    """Tests for text extraction from PDF files."""

    def test_extract_text_from_valid_pdf(self):
        """Should extract non-empty text from a valid PDF."""
        pdf_path = _find_test_pdf()
        text = extract_text(pdf_path)
        assert text, "Should extract some text"
        assert isinstance(text, str)
        assert len(text.strip()) > 50, "Should extract substantial text"

    def test_extract_text_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            extract_text("nonexistent_file.pdf")

    def test_extract_text_handles_empty_pages(self):
        """Should handle pages with no text gracefully."""
        pdf_path = _find_test_pdf()
        text = extract_text(pdf_path)
        # Should not crash, even if some pages have no text
        assert isinstance(text, str)

    def test_extract_text_returns_chinese(self):
        """Should extract Chinese characters correctly."""
        pdf_path = _find_test_pdf()
        text = extract_text(pdf_path)
        # Chinese policy documents should contain CJK characters
        has_chinese = any('一' <= c <= '鿿' for c in text)
        assert has_chinese, "Policy PDF should contain Chinese text"


class TestExtractMetadata:
    """Tests for PDF metadata extraction."""

    def test_extract_metadata_returns_dict(self):
        """Should return a dictionary with expected keys."""
        pdf_path = _find_test_pdf()
        meta = extract_metadata(pdf_path)
        assert isinstance(meta, dict)
        assert "title" in meta
        assert "page_count" in meta
        assert "file_size_bytes" in meta

    def test_extract_metadata_page_count_positive(self):
        """Page count should be > 0 for valid PDF."""
        pdf_path = _find_test_pdf()
        meta = extract_metadata(pdf_path)
        assert meta["page_count"] > 0

    def test_extract_metadata_file_size_bytes_positive(self):
        """File size should be > 0."""
        pdf_path = _find_test_pdf()
        meta = extract_metadata(pdf_path)
        assert meta["file_size_bytes"] > 0


class TestNeedsOCR:
    """Tests for OCR detection."""

    def test_needs_ocr_on_text_pdf(self):
        """Should return False for PDF with extractable text."""
        pdf_path = _find_test_pdf()
        result = needs_ocr(pdf_path)
        assert result is False, "PDF with text should not need OCR"

    def test_needs_ocr_file_not_found(self):
        """Should raise error for missing file."""
        with pytest.raises(FileNotFoundError):
            needs_ocr("nonexistent_file.pdf")


class TestExtractTextPerPage:
    """Tests for page-level text extraction."""

    def test_extract_text_per_page(self):
        """Should return list of page dicts."""
        pdf_path = _find_test_pdf()
        pages = extract_text_per_page(pdf_path)
        assert isinstance(pages, list)
        if pages:
            assert "page_num" in pages[0]
            assert "text" in pages[0]


class TestChunkTextBackwardCompat:
    """Tests for backward-compatible chunk_text."""

    def test_chunk_text_returns_list(self):
        """Should return list of chunk dicts."""
        text = "这是一个测试段落。它包含了一些中文内容，用于测试分段功能。" * 50
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        if chunks:
            assert "id" in chunks[0]
            assert "content" in chunks[0]
            assert "metadata" in chunks[0]

    def test_chunk_text_empty_input(self):
        """Should return empty list for empty input."""
        chunks = chunk_text("")
        assert chunks == []


class TestExtractTextWithOCR:
    """Tests for OCR-enabled text extraction."""

    def test_extract_text_with_ocr_on_text_pdf(self):
        """Should work same as extract_text for non-scanned PDFs."""
        pdf_path = _find_test_pdf()
        text = extract_text_with_ocr(pdf_path)
        assert isinstance(text, str)
        assert len(text.strip()) > 0

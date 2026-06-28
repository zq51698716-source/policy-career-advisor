"""
Integration test — full RAG pipeline: parse → chunk → embed → store → search.

NOTE: Requires a real PDF file in the uploads/ directory.
Requires the embedding model to be downloaded (~24MB first run).
Skip with SKIP_INTEGRATION_TESTS=1.
"""

import os
import sys
import shutil
import tempfile
import pytest
from pathlib import Path

# Skip integration tests if explicitly disabled
SKIP_INTEGRATION = os.getenv("SKIP_INTEGRATION_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    SKIP_INTEGRATION,
    reason="SKIP_INTEGRATION_TESTS=1 — skipping integration tests"
)

TESTS_DIR = Path(__file__).parent.parent
UPLOADS_DIR = TESTS_DIR.parent / "uploads"


def _find_test_pdf():
    """Find a real PDF file for integration testing."""
    if UPLOADS_DIR.exists():
        for f in UPLOADS_DIR.iterdir():
            if f.suffix.lower() == ".pdf":
                return f
    pytest.skip("No test PDF found in uploads/ — add a PDF for integration tests")


class TestRAGPipeline:
    """End-to-end RAG pipeline tests."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset state before each test."""
        from backend.rag.vector_store import reset_store
        from backend.rag.embedder import EmbeddingModel
        reset_store()
        yield
        reset_store()

    def test_parse_chunk_store_search(self):
        """
        Full pipeline: PDF text → chunk → embed → store → search.

        Verifies that all modules work together correctly.
        """
        from backend.rag.pdf_parser import extract_text, extract_metadata
        from backend.rag.chunker import SemanticChunker
        from backend.rag.vector_store import add_document, search, list_documents, delete_document

        pdf_path = str(_find_test_pdf())

        # Step 1: Parse PDF
        text = extract_text(pdf_path)
        assert text, "Should extract text from PDF"
        assert len(text.strip()) > 100, "Should get substantial text"

        meta = extract_metadata(pdf_path)
        assert meta["page_count"] > 0

        # Step 2: Semantic chunking
        chunker = SemanticChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk(text, filename=_find_test_pdf().name)
        assert len(chunks) > 0
        for c in chunks:
            assert c["content"].strip()
            assert c["metadata"]["filename"] == _find_test_pdf().name

        # Step 3 & 4: Embed and store
        doc_id = add_document(_find_test_pdf().name, chunks)
        assert doc_id
        assert len(doc_id) == 8

        # Verify document is listed
        docs = list_documents()
        assert any(d["doc_id"] == doc_id for d in docs)

        # Step 5: Search
        # Use a query related to the document's likely topic
        results = search("人才", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0

        for r in results:
            assert "content" in r
            assert "filename" in r
            assert "doc_id" in r
            assert "score" in r
            assert r["score"] >= 0

        # Clean up
        ok = delete_document(doc_id)
        assert ok is True

    def test_multiple_documents(self):
        """Upload same PDF twice and verify dedup in search."""
        from backend.rag.pdf_parser import extract_text
        from backend.rag.chunker import SemanticChunker
        from backend.rag.vector_store import add_document, search, delete_document, list_documents

        pdf_path = str(_find_test_pdf())
        text = extract_text(pdf_path)
        chunker = SemanticChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk(text, filename=_find_test_pdf().name)

        # Add same document twice
        doc1 = add_document(_find_test_pdf().name, chunks)
        doc2 = add_document(_find_test_pdf().name, chunks)

        docs = list_documents()
        assert len(docs) >= 2

        # Search should return results from both
        results = search("政策", top_k=3)
        # Results should be deduplicated by doc_id
        doc_ids = [r["doc_id"] for r in results]
        assert len(doc_ids) == len(set(doc_ids)), "Should not return duplicate doc_ids"

        # Clean up
        delete_document(doc1)
        delete_document(doc2)

    def test_search_modes(self):
        """Test all three search modes (vector, bm25, hybrid)."""
        from backend.rag.pdf_parser import extract_text
        from backend.rag.chunker import SemanticChunker
        from backend.rag.vector_store import add_document, search, delete_document

        pdf_path = str(_find_test_pdf())
        text = extract_text(pdf_path)
        chunker = SemanticChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk(text, filename=_find_test_pdf().name)
        doc_id = add_document(_find_test_pdf().name, chunks)

        for mode in ["vector", "bm25", "hybrid"]:
            results = search("人才政策", top_k=3, search_mode=mode)
            assert isinstance(results, list), f"Mode '{mode}' should return a list"
            if results:
                # Check result format
                assert "content" in results[0], f"Mode '{mode}' results missing 'content'"
                assert "score" in results[0], f"Mode '{mode}' results missing 'score'"

        delete_document(doc_id)

    def test_search_empty_store(self):
        """Search on empty store should return empty list."""
        from backend.rag.vector_store import search
        results = search("测试查询", top_k=5)
        assert results == []

"""
Unit tests for vector_store — ChromaDB-backed vector storage and search.
"""

import os
import pytest
import shutil
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np


# We need to mock the embedding model to avoid loading the real model
@pytest.fixture(scope="module")
def mock_embedder():
    """Create a mock embedding model that returns fake vectors."""
    with patch("backend.rag.vector_store.EmbeddingModel") as MockEmbedder:
        instance = MockEmbedder.get_instance.return_value
        instance.dimension = 512
        instance.encode.return_value = np.random.randn(5, 512).astype(np.float32)
        instance.encode.return_value /= np.linalg.norm(
            instance.encode.return_value, axis=1, keepdims=True
        )
        instance.encode_query.return_value = np.random.randn(512).astype(np.float32)
        instance.encode_query.return_value /= np.linalg.norm(
            instance.encode_query.return_value
        )
        yield instance


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary ChromaDB directory for isolated testing."""
    tmpdir = tempfile.mkdtemp(prefix="chroma_test_")
    with patch("backend.rag.vector_store.CHROMA_DATA_DIR", tmpdir):
        # Also need to reset the singleton
        from backend.rag import vector_store
        vector_store.reset_store()
        yield tmpdir
        vector_store.reset_store()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_chunks():
    """Create sample chunk dicts for testing."""
    return [
        {
            "id": "chunk_0",
            "content": "深圳市人才引进实施办法规定，高层次人才可享受住房补贴和生活补贴。",
            "metadata": {
                "chunk_index": 0,
                "total_chunks": 3,
                "filename": "深圳人才政策2026.pdf",
                "heading": "第一章 总则",
                "char_count": 35,
            },
        },
        {
            "id": "chunk_1",
            "content": "申请人才住房补贴需提交身份证、学历证书、劳动合同等材料。",
            "metadata": {
                "chunk_index": 1,
                "total_chunks": 3,
                "filename": "深圳人才政策2026.pdf",
                "heading": "",
                "char_count": 30,
            },
        },
        {
            "id": "chunk_2",
            "content": "高校毕业生就业扶持政策包括社保补贴、创业担保贷款和见习补贴。",
            "metadata": {
                "chunk_index": 2,
                "total_chunks": 3,
                "filename": "就业扶持政策.pdf",
                "heading": "第三条 就业扶持",
                "char_count": 33,
            },
        },
    ]


class TestVectorStore:
    """Tests for VectorStore CRUD operations with mocked embeddings."""

    def test_init_creates_collection(self, temp_chroma_dir, mock_embedder):
        """VectorStore should create a ChromaDB collection on init."""
        from backend.rag.vector_store import VectorStore
        store = VectorStore()
        assert store.count == 0
        assert store.collection is not None

    def test_add_document(self, temp_chroma_dir, mock_embedder, sample_chunks):
        """add_document should incrementally insert chunks."""
        from backend.rag.vector_store import VectorStore
        store = VectorStore()

        mock_embedder.encode.return_value = np.random.randn(
            len(sample_chunks), 512
        ).astype(np.float32)
        # Normalize
        vecs = mock_embedder.encode.return_value
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        doc_id = store.add_document("test_policy.pdf", sample_chunks)
        assert doc_id is not None
        assert len(doc_id) == 8  # UUID hex[:8]
        assert store.count == len(sample_chunks)

    def test_add_document_empty_chunks(self, temp_chroma_dir, mock_embedder):
        """add_document should raise ValueError for empty chunk list."""
        from backend.rag.vector_store import VectorStore
        store = VectorStore()
        with pytest.raises(ValueError):
            store.add_document("empty.pdf", [])

    def test_search_empty_store(self, temp_chroma_dir, mock_embedder):
        """Search on empty store should return empty list."""
        from backend.rag.vector_store import VectorStore
        store = VectorStore()
        results = store.search("人才住房补贴")
        assert results == []

    def test_search_returns_results(self, temp_chroma_dir, mock_embedder, sample_chunks):
        """Search on populated store should return results."""
        from backend.rag.vector_store import VectorStore
        store = VectorStore()

        # Configure mock for encode (add)
        mock_embedder.encode.return_value = np.random.randn(
            len(sample_chunks), 512
        ).astype(np.float32)
        vecs = mock_embedder.encode.return_value
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        store.add_document("test_policy.pdf", sample_chunks)

        # Configure mock for encode_query (search)
        mock_embedder.encode_query.return_value = np.random.randn(512).astype(np.float32)
        mock_embedder.encode_query.return_value /= np.linalg.norm(
            mock_embedder.encode_query.return_value
        )

        # Now search - will use ChromaDB's built-in query
        results = store.search("人才住房补贴", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0

        # Validate result format
        for r in results:
            assert "content" in r
            assert "filename" in r
            assert "doc_id" in r
            assert "score" in r

    def test_list_documents(self, temp_chroma_dir, mock_embedder, sample_chunks):
        """list_documents should return unique documents."""
        from backend.rag.vector_store import VectorStore
        store = VectorStore()

        mock_embedder.encode.return_value = np.random.randn(
            len(sample_chunks), 512
        ).astype(np.float32)
        vecs = mock_embedder.encode.return_value
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        store.add_document("test_policy.pdf", sample_chunks)
        docs = store.list_documents()
        assert isinstance(docs, list)
        assert len(docs) >= 1

        for d in docs:
            assert "doc_id" in d
            assert "filename" in d
            assert "chunk_count" in d

    def test_delete_document(self, temp_chroma_dir, mock_embedder, sample_chunks):
        """delete_document should remove all chunks and return True."""
        from backend.rag.vector_store import VectorStore
        store = VectorStore()

        mock_embedder.encode.return_value = np.random.randn(
            len(sample_chunks), 512
        ).astype(np.float32)
        vecs = mock_embedder.encode.return_value
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        doc_id = store.add_document("test_policy.pdf", sample_chunks)
        assert store.count > 0

        ok = store.delete_document(doc_id)
        assert ok is True
        assert store.count == 0

    def test_delete_nonexistent_document(self, temp_chroma_dir, mock_embedder):
        """delete_document should return False for nonexistent doc_id."""
        from backend.rag.vector_store import VectorStore
        store = VectorStore()
        ok = store.delete_document("nonexistent")
        assert ok is False

    def test_health_check(self, temp_chroma_dir, mock_embedder):
        """health_check should return status dict."""
        from backend.rag.vector_store import VectorStore
        store = VectorStore()
        health = store.health_check()
        assert health["status"] == "healthy"
        assert "chunk_count" in health
        assert "embedding_dimension" in health


class TestPublicAPI:
    """Tests for module-level convenience functions."""

    def test_get_chunk_count(self, temp_chroma_dir, mock_embedder):
        """Module-level get_chunk_count should work."""
        from backend.rag import vector_store
        vector_store.reset_store()
        count = vector_store.get_chunk_count()
        assert isinstance(count, int)

    def test_health_check(self, temp_chroma_dir, mock_embedder):
        """Module-level health_check should work."""
        from backend.rag import vector_store
        vector_store.reset_store()
        health = vector_store.health_check()
        assert health["status"] == "healthy"

"""
Unit tests for hybrid_search — BM25 + vector with RRF fusion.

These tests use mock data and a mock embedder; no real models needed.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from backend.rag.hybrid_search import HybridSearcher, RRF_K


@pytest.fixture
def mock_embedder():
    """Mock embedder returning predictable vectors."""
    emb = MagicMock()
    emb.encode_query.return_value = np.array([1.0, 0.0, 0.0, 0.0])  # 4D for simplicity
    emb.encode_query.return_value /= np.linalg.norm(emb.encode_query.return_value)
    emb.dimension = 4
    return emb


@pytest.fixture
def sample_data():
    """Create sample document data for hybrid search testing."""
    ids = [
        "doc1_chunk_0",
        "doc1_chunk_1",
        "doc2_chunk_0",
        "doc3_chunk_0",
    ]
    documents = [
        "深圳市人才住房补贴政策规定，高层次人才可申请免租住房或租房补贴。",
        "申请人才补贴需提交以下材料：身份证、学历证、劳动合同。",
        "高校毕业生创业补贴政策：符合条件的可获一次性创业资助。",
        "今天天气很好，适合户外运动和野餐聚会。",
    ]
    metadatas = [
        {"doc_id": "doc1", "filename": "人才住房补贴政策.pdf", "heading": "第一章"},
        {"doc_id": "doc1", "filename": "人才住房补贴政策.pdf", "heading": ""},
        {"doc_id": "doc2", "filename": "创业补贴政策.pdf", "heading": "第三条"},
        {"doc_id": "doc3", "filename": "无关文档.pdf", "heading": ""},
    ]
    # Mock embeddings: doc0 is close to query [1,0,0,0], doc1 similar, doc2 somewhat, doc3 orthogonal
    embeddings = [
        [0.95, 0.1, 0.0, 0.0],
        [0.8, 0.2, 0.1, 0.0],
        [0.3, 0.5, 0.3, 0.1],
        [0.0, 0.0, 0.0, 1.0],
    ]
    # Normalize
    embeddings = (np.array(embeddings) / np.linalg.norm(embeddings, axis=1, keepdims=True)).tolist()

    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "embeddings": embeddings,
    }


class TestHybridSearcher:
    """Tests for HybridSearcher."""

    def test_init_builds_bm25(self, sample_data, mock_embedder):
        """Initialization should build BM25 index from documents."""
        searcher = HybridSearcher(
            ids=sample_data["ids"],
            documents=sample_data["documents"],
            metadatas=sample_data["metadatas"],
            embeddings=sample_data["embeddings"],
            embedder=mock_embedder,
        )
        assert searcher._bm25 is not None
        assert len(searcher._documents) == 4

    def test_init_empty(self, mock_embedder):
        """Should handle empty document list."""
        searcher = HybridSearcher(
            ids=[], documents=[], metadatas=[], embeddings=[], embedder=mock_embedder
        )
        assert searcher._bm25 is None
        results = searcher.search("测试")
        assert results == []

    def test_bm25_finds_keyword_match(self, sample_data, mock_embedder):
        """BM25 should find documents with matching keywords."""
        searcher = HybridSearcher(
            ids=sample_data["ids"],
            documents=sample_data["documents"],
            metadatas=sample_data["metadatas"],
            embeddings=sample_data["embeddings"],
            embedder=mock_embedder,
        )
        results = searcher.search_bm25("人才住房补贴", top_k=2)
        assert len(results) > 0
        # The top result should contain "人才住房补贴" in its content
        assert "人才住房补贴" in results[0]["content"]
        assert results[0]["source"] == "bm25"

    def test_vector_finds_semantic_match(self, sample_data, mock_embedder):
        """Vector search should find semantically similar documents."""
        searcher = HybridSearcher(
            ids=sample_data["ids"],
            documents=sample_data["documents"],
            metadatas=sample_data["metadatas"],
            embeddings=sample_data["embeddings"],
            embedder=mock_embedder,
        )
        results = searcher.search_vector("查询", top_k=3)
        assert len(results) > 0
        assert results[0]["source"] == "vector"
        # Top result should have highest cosine similarity
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]

    def test_hybrid_search_returns_results(self, sample_data, mock_embedder):
        """Hybrid search should return combined and deduplicated results."""
        searcher = HybridSearcher(
            ids=sample_data["ids"],
            documents=sample_data["documents"],
            metadatas=sample_data["metadatas"],
            embeddings=sample_data["embeddings"],
            embedder=mock_embedder,
        )
        results = searcher.search("人才住房补贴政策", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3
        # Results should have the required keys
        for r in results:
            assert "content" in r
            assert "score" in r
            assert "doc_id" in r
            assert "source" in r  # e.g., "bm25+vector"

    def test_rrf_fusion_deduplicates(self, sample_data, mock_embedder):
        """RRF fusion should deduplicate by document ID."""
        searcher = HybridSearcher(
            ids=sample_data["ids"],
            documents=sample_data["documents"],
            metadatas=sample_data["metadatas"],
            embeddings=sample_data["embeddings"],
            embedder=mock_embedder,
        )
        results = searcher.search("人才住房补贴", top_k=3)
        # Should not return same doc_id twice
        doc_ids = [r["doc_id"] for r in results]
        assert len(doc_ids) == len(set(doc_ids)), f"Duplicates found: {doc_ids}"

    def test_hybrid_search_ranks_relevant_higher(self, sample_data, mock_embedder):
        """Relevant documents should rank above irrelevant ones."""
        searcher = HybridSearcher(
            ids=sample_data["ids"],
            documents=sample_data["documents"],
            metadatas=sample_data["metadatas"],
            embeddings=sample_data["embeddings"],
            embedder=mock_embedder,
        )
        results = searcher.search("人才住房补贴政策", top_k=4)
        if len(results) >= 2:
            # The irrelevant doc should NOT be ranked first
            top_content = results[0]["content"]
            assert "人才" in top_content or "住房" in top_content, (
                f"Expected relevant doc first, got: {top_content[:50]}..."
            )

    def test_search_empty_string(self, sample_data, mock_embedder):
        """Should handle empty query gracefully."""
        searcher = HybridSearcher(
            ids=sample_data["ids"],
            documents=sample_data["documents"],
            metadatas=sample_data["metadatas"],
            embeddings=sample_data["embeddings"],
            embedder=mock_embedder,
        )
        results = searcher.search("", top_k=3)
        # BM25 gives zero scores for empty query, vector may also fail
        # Should not crash, may return empty or low-score results
        assert isinstance(results, list)

"""
Unit tests for reranker — optional cross-encoder reranking.

Reranker is disabled by default (RERANKER_ENABLED=false).
These tests verify the opt-in flow with mocking.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from backend.rag.reranker import Reranker, rerank_if_enabled


@pytest.fixture
def candidate_docs():
    """Sample search results for reranking."""
    return [
        {
            "id": "doc1_chunk_0",
            "content": "深圳市人才住房补贴政策：高层次人才可申请租房补贴，每月最高5000元。",
            "filename": "人才政策.pdf",
            "score": 0.85,
            "source": "hybrid",
        },
        {
            "id": "doc2_chunk_1",
            "content": "申请人才住房补贴需准备以下材料：身份证复印件、学历证明、社保证明。",
            "filename": "人才政策.pdf",
            "score": 0.72,
            "source": "vector",
        },
        {
            "id": "doc3_chunk_0",
            "content": "高校毕业生就业补贴申请流程：网上申报、审核、公示、发放。",
            "filename": "就业政策.pdf",
            "score": 0.60,
            "source": "bm25",
        },
    ]


class TestRerankerWhenDisabled:
    """Tests when reranker is disabled (default)."""

    def test_get_instance_returns_none(self):
        """When RERANKER_ENABLED=false, get_instance() returns None."""
        with patch("backend.rag.reranker.RERANKER_ENABLED", False):
            Reranker._instance = None
            result = Reranker.get_instance()
            assert result is None

    def test_is_available_returns_false(self):
        """is_available() should return False when disabled."""
        with patch("backend.rag.reranker.RERANKER_ENABLED", False):
            Reranker._instance = None
            assert Reranker.is_available() is False

    def test_rerank_if_enabled_passthrough(self, candidate_docs):
        """rerank_if_enabled should return input unchanged when disabled."""
        with patch("backend.rag.reranker.RERANKER_ENABLED", False):
            result = rerank_if_enabled("人才住房补贴", candidate_docs, top_k=2)
            assert result == candidate_docs[:2]  # Simply truncated

    def test_rerank_if_enabled_empty_list(self):
        """Should handle empty candidate list."""
        with patch("backend.rag.reranker.RERANKER_ENABLED", False):
            result = rerank_if_enabled("查询", [], top_k=3)
            assert result == []


class TestRerankerWhenEnabled:
    """Tests when reranker is enabled (mock the model to avoid loading)."""

    def test_rerank_updates_scores(self, candidate_docs):
        """Reranker should add rerank_score and re-sort results."""
        mock_model = MagicMock()
        # Simulate the reranker giving different scores
        mock_model.predict.return_value = [0.9, 0.4, 0.7]

        with patch("backend.rag.reranker.RERANKER_ENABLED", True):
            with patch("backend.rag.reranker.Reranker._load") as mock_load:
                Reranker._instance = None
                reranker = Reranker("test-model")
                reranker._model = mock_model

                result = reranker.rerank("人才住房补贴申请流程", candidate_docs, top_k=3)

                assert len(result) == 3
                # Results should be re-sorted by rerank_score (0.9, 0.7, 0.4)
                assert result[0]["rerank_score"] == 0.9
                assert result[1]["rerank_score"] == 0.7
                assert result[2]["rerank_score"] == 0.4
                # Original score preserved
                assert "score" in result[0]

    def test_rerank_top_k_truncation(self, candidate_docs):
        """Reranker should return at most top_k results."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7]

        with patch("backend.rag.reranker.RERANKER_ENABLED", True):
            with patch("backend.rag.reranker.Reranker._load"):
                Reranker._instance = None
                reranker = Reranker("test-model")
                reranker._model = mock_model

                result = reranker.rerank("查询", candidate_docs, top_k=2)
                assert len(result) == 2

    def test_reranker_load_failure_graceful(self, candidate_docs):
        """If model fails to load, _model should be None — no crash."""
        with patch("backend.rag.reranker.RERANKER_ENABLED", True):
            with patch("backend.rag.reranker.Reranker._load") as mock_load:
                mock_load.side_effect = Exception("Network error")
                Reranker._instance = None
                reranker = Reranker("test-model")
                # _load is called in __init__, but we mocked it to raise
                # In reality the error is caught; test that _model remains None
                reranker._model = None  # Simulate failed load
                result = reranker.rerank("查询", candidate_docs, top_k=3)
                # Should return original candidates (pass-through on failure)
                assert len(result) == 3

    def test_rerank_if_enabled_with_reranker(self, candidate_docs):
        """rerank_if_enabled with working reranker should rerank."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.3, 0.9, 0.5]

        with patch("backend.rag.reranker.RERANKER_ENABLED", True):
            with patch("backend.rag.reranker.Reranker._load"):
                Reranker._instance = None
                reranker = Reranker("test-model")
                reranker._model = mock_model
                Reranker._instance = reranker

                result = rerank_if_enabled("查询", candidate_docs, top_k=2)
                assert len(result) == 2
                # Top should be the one with highest rerank_score
                assert result[0]["rerank_score"] == 0.9

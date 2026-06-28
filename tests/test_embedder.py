"""
Unit tests for embedder — EmbeddingModel with BGE-small-zh-v1.5.

NOTE: These tests download the model on first run (~24MB).
If the test environment has no internet access, set SKIP_EMBEDDING_TESTS=1.
"""

import os
import pytest
import numpy as np

from backend.rag.embedder import (
    EmbeddingModel,
    encode_documents,
    encode_query,
    get_embedding_dimension,
    BGE_QUERY_INSTRUCTION,
)


# Skip embedding tests if explicitly disabled (CI without internet)
SKIP_EMBEDDING = os.getenv("SKIP_EMBEDDING_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    SKIP_EMBEDDING,
    reason="SKIP_EMBEDDING_TESTS=1 — skipping embedding model tests"
)


@pytest.fixture(scope="module")
def model():
    """Load the embedding model once for all tests in this module."""
    EmbeddingModel.reset_instance()
    model = EmbeddingModel.get_instance()
    yield model
    EmbeddingModel.reset_instance()


class TestEmbeddingModel:
    """Tests for the EmbeddingModel singleton."""

    def test_model_is_singleton(self, model):
        """get_instance() should always return the same instance."""
        m2 = EmbeddingModel.get_instance()
        assert model is m2

    def test_model_has_dimension(self, model):
        """Model should report a positive embedding dimension."""
        dim = model.dimension
        assert dim > 0
        assert isinstance(dim, int)
        # BGE-small-zh-v1.5 has 512 dimensions
        assert dim == 512

    def test_encode_returns_numpy_array(self, model):
        """encode() should return a numpy array of correct shape."""
        texts = ["深圳市人才补贴政策", "高校毕业生就业扶持办法"]
        result = model.encode(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, model.dimension)

    def test_encode_normalized(self, model):
        """Output vectors should be L2-normalized (magnitude ≈ 1.0)."""
        texts = ["测试文本"]
        result = model.encode(texts)
        magnitude = np.linalg.norm(result[0])
        assert abs(magnitude - 1.0) < 0.01, f"Vector not normalized: norm={magnitude}"

    def test_encode_single_text(self, model):
        """Encoding a single text should return shape (1, dim)."""
        result = model.encode(["单个文本"])
        assert result.shape == (1, model.dimension)

    def test_encode_empty_list(self, model):
        """Encoding empty list should return empty array."""
        result = model.encode([])
        assert result.size == 0

    def test_encode_query_returns_vector(self, model):
        """encode_query should return a 1D vector."""
        result = model.encode_query("人才住房补贴")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) == model.dimension

    def test_encode_query_is_normalized(self, model):
        """Query vector should also be L2-normalized."""
        result = model.encode_query("人才住房补贴")
        magnitude = np.linalg.norm(result)
        assert abs(magnitude - 1.0) < 0.01

    def test_encode_query_vs_encode_different(self, model):
        """For BGE models, encode_query should differ from encode due to instruction prefix."""
        query = "人才住房补贴怎么申请"
        doc_vec = model.encode([query])[0]
        query_vec = model.encode_query(query)
        # They should be different because query gets the instruction prefix
        cos_sim = np.dot(doc_vec, query_vec)
        # They might be similar but not identical
        assert cos_sim > 0.5  # Still semantically related
        assert not np.allclose(doc_vec, query_vec, atol=1e-4)

    def test_batch_encode_consistent(self, model):
        """Batch encoding should produce same results as individual encoding."""
        texts = ["政策A", "政策B", "政策C"]
        batch_result = model.encode(texts)
        individual_results = [model.encode([t])[0] for t in texts]
        for i, ind in enumerate(individual_results):
            assert np.allclose(batch_result[i], ind, atol=1e-5)

    def test_semantic_similarity(self, model):
        """Semantically similar texts should have higher cosine similarity."""
        query_vec = model.encode_query("人才住房补贴")
        similar_text = model.encode(["深圳市人才安居工程住房补贴政策"])[0]
        unrelated_text = model.encode(["今天天气很好适合出去游玩"])[0]

        sim_similar = float(np.dot(query_vec, similar_text))
        sim_unrelated = float(np.dot(query_vec, unrelated_text))

        assert sim_similar > sim_unrelated, (
            f"Similar text (score={sim_similar:.3f}) should score higher "
            f"than unrelated text (score={sim_unrelated:.3f})"
        )


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_encode_documents(self, model):
        """Module-level encode_documents should work."""
        result = encode_documents(["测试文本"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, model.dimension)

    def test_encode_query(self, model):
        """Module-level encode_query should work."""
        result = encode_query("测试查询")
        assert isinstance(result, np.ndarray)
        assert len(result) == model.dimension

    def test_get_embedding_dimension(self, model):
        """get_embedding_dimension should return correct value."""
        dim = get_embedding_dimension()
        assert dim == model.dimension

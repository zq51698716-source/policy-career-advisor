"""
Cross-encoder reranker — refines retrieval results with a more precise
(but slower) model. Optional module, disabled by default.

When enabled (RERANKER_ENABLED=true), reranks the top-N candidates from
hybrid search before returning final results.

Default model: BAAI/bge-reranker-v2-m3 (multilingual, ~568M params)
Alternative: BAAI/bge-reranker-base (~278M params, faster)

IMPORTANT: Reranker models require ~1-2GB disk space and significant CPU/GPU
memory. Only enable if retrieval precision is critical and hardware permits.
"""

import logging
from typing import Optional

from backend.config import RERANKER_ENABLED

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
LIGHT_RERANKER_MODEL = "BAAI/bge-reranker-base"  # Smaller alternative


class Reranker:
    """
    Cross-encoder reranker for improving retrieval precision.

    Loads a cross-encoder model that jointly encodes the (query, document)
    pair for more accurate relevance scoring than bi-encoder embeddings alone.

    Usage:
        if RERANKER_ENABLED:
            reranker = Reranker()
            refined = reranker.rerank(query, candidates, top_k=3)
    """

    _instance: Optional["Reranker"] = None

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self._model = None
        if RERANKER_ENABLED:
            try:
                self._load()
            except Exception as e:
                logger.warning(f"Reranker failed to load: {e} — reranking disabled")
                self._model = None

    def _load(self):
        """Load the cross-encoder model (lazy, on first use)."""
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,  # Limit input length for speed
                trust_remote_code=False,
            )
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model '{self.model_name}': {e}")
            logger.warning("Reranker disabled — continuing without reranking")
            self._model = None

    @classmethod
    def get_instance(cls) -> Optional["Reranker"]:
        """
        Get or create the global reranker instance.

        Returns None if RERANKER_ENABLED is False.
        """
        if not RERANKER_ENABLED:
            return None
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def is_available(cls) -> bool:
        """Check if reranker is enabled and loaded successfully."""
        instance = cls.get_instance()
        return instance is not None and instance._model is not None

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank candidate documents using the cross-encoder.

        Args:
            query: Search query.
            candidates: List of result dicts with "content" key.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked candidates with updated scores.
        """
        if not self._model or not candidates:
            return candidates[:top_k]

        # Prepare (query, document) pairs
        pairs = [(query, c["content"]) for c in candidates]

        try:
            scores = self._model.predict(
                pairs,
                show_progress_bar=False,
                batch_size=8,
            )

            # Attach scores and sort
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = round(float(score), 4)

            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            logger.debug(
                f"Reranked {len(candidates)} candidates, "
                f"top score: {candidates[0].get('rerank_score', 0):.4f}"
            )

        except Exception as e:
            logger.error(f"Reranking failed: {e} — returning original order")
            # Fall through to return original candidates

        return candidates[:top_k]


def rerank_if_enabled(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Convenience function: rerank if the reranker is configured, otherwise pass-through.

    This is the primary entry point for the RAG pipeline.
    """
    if not RERANKER_ENABLED:
        return candidates[:top_k]

    reranker = Reranker.get_instance()
    if reranker is None or not reranker._model:
        return candidates[:top_k]

    return reranker.rerank(query, candidates, top_k)

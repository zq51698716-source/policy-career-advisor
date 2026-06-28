"""
Embedding model wrapper — provides a singleton interface to sentence-transformers
for Chinese text vectorization.

Default model: BAAI/bge-small-zh-v1.5 (24MB, CPU-friendly, top-3 on C-MTEB leaderboard)
Configurable via EMBEDDING_MODEL env var.

BGE models require an instruction prefix for queries. We apply:
  "为这个句子生成表示以用于检索相关文章：" for queries
  "" (no prefix) for documents
"""

import logging
from typing import Optional

import numpy as np

from backend.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# BGE models use this instruction prefix for asymmetric embeddings
# Reference: https://huggingface.co/BAAI/bge-small-zh-v1.5
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# Models known to benefit from query instruction prefix
BGE_MODEL_PREFIXES = {
    "BAAI/bge-small-zh": BGE_QUERY_INSTRUCTION,
    "BAAI/bge-small-zh-v1.5": BGE_QUERY_INSTRUCTION,
    "BAAI/bge-base-zh": BGE_QUERY_INSTRUCTION,
    "BAAI/bge-base-zh-v1.5": BGE_QUERY_INSTRUCTION,
    "BAAI/bge-large-zh": BGE_QUERY_INSTRUCTION,
    "BAAI/bge-large-zh-v1.5": BGE_QUERY_INSTRUCTION,
    "BAAI/bge-m3": "",
    "moka-ai/m3e-small": "",
    "moka-ai/m3e-base": "",
    "moka-ai/m3e-large": "",
}


class EmbeddingModel:
    """
    Thread-safe singleton embedding model.

    Lazy-loads on first call — no GPU/CPU cost at import time.
    Caches the model in memory for subsequent calls.

    Usage:
        model = EmbeddingModel.get_instance()
        doc_vecs = model.encode(["政策文本1", "政策文本2"])
        query_vec = model.encode_query("人才住房补贴")
    """

    _instance: Optional["EmbeddingModel"] = None
    _model = None

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._dimension: Optional[int] = None
        self._load()

    def _load(self):
        """Load the sentence-transformers model."""
        import sys as _sys
        import os as _os

        # Force-reload huggingface_hub if HF_ENDPOINT was set after initial import.
        # This can happen when chromadb or another dependency imports huggingface_hub
        # before config.py gets a chance to set the env var.
        _hf_module = _sys.modules.get("huggingface_hub")
        if _hf_module and "constants" in _sys.modules:
            import importlib
            importlib.reload(_sys.modules["huggingface_hub.constants"])
            # Also re-import endpoint-dependent submodules
            for _name in list(_sys.modules):
                if _name.startswith("huggingface_hub."):
                    try:
                        importlib.reload(_sys.modules[_name])
                    except Exception:
                        pass

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=False,
            )
            self._dimension = self._model.get_embedding_dimension()
            logger.info(
                f"Embedding model loaded. Dimension: {self._dimension}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
            hint = ""
            if "10060" in str(e) or "getaddrinfo" in str(e):
                hint = (
                    "\nHint: HuggingFace Hub may be inaccessible from your network. "
                    "Set HF_ENDPOINT=https://hf-mirror.com in your .env file, or:\n"
                    "  set HF_ENDPOINT=https://hf-mirror.com\n"
                    "Or download the model manually to "
                    "~/.cache/huggingface/hub/"
                )
            raise RuntimeError(
                f"Cannot load embedding model '{self.model_name}'. "
                f"Ensure it is available on HuggingFace Hub and network is accessible. "
                f"Set EMBEDDING_MODEL env var to switch models."
                f"{hint}"
            ) from e

    @classmethod
    def get_instance(cls) -> "EmbeddingModel":
        """Get or create the global singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (useful for testing with different models)."""
        cls._instance = None

    @property
    def dimension(self) -> int:
        """Output vector dimension."""
        if self._dimension is None:
            raise RuntimeError("Model not loaded yet")
        return self._dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode documents into embeddings.

        Documents are NOT prefixed with instructions (BGE convention).

        Args:
            texts: List of document texts.
            batch_size: Batch size for encoding.
            show_progress_bar: Whether to show tqdm progress bar.

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a search query.

        For BGE models, prepends the instruction prefix.
        For other models (M3E, etc.), encodes directly.

        Args:
            query: Search query string.

        Returns:
            numpy array of shape (dimension,)
        """
        instruction = BGE_MODEL_PREFIXES.get(self.model_name, "")
        if instruction:
            query = f"{instruction}{query}"

        return self._model.encode(
            [query],
            normalize_embeddings=True,
        )[0]


# Convenience functions that operate on the singleton
def _get_model() -> EmbeddingModel:
    return EmbeddingModel.get_instance()


def encode_documents(texts: list[str]) -> np.ndarray:
    """Encode documents to embeddings using the global singleton."""
    return _get_model().encode(texts)


def encode_query(query: str) -> np.ndarray:
    """Encode a query to embedding using the global singleton."""
    return _get_model().encode_query(query)


def get_embedding_dimension() -> int:
    """Get the embedding dimension of the current model."""
    return _get_model().dimension

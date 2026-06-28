"""
Hybrid search — combines BM25 (keyword) and vector (semantic) retrieval
using Reciprocal Rank Fusion (RRF).

RRF formula: score(d) = sum( 1 / (k + rank_i(d)) ) for each ranking i
where k=60 is the standard smoothing constant.

This provides robust retrieval that works well for both exact keyword
matches AND semantic understanding — critical for Chinese policy documents
where terminology can vary widely.
"""

import logging
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from backend.config import RAG_TOP_K

logger = logging.getLogger(__name__)

# RRF smoothing constant (standard value from literature)
RRF_K = 60


class HybridSearcher:
    """
    Performs hybrid BM25 + vector search with RRF fusion.

    Maintains a BM25 index over the document corpus and delegates
    vector search to the EmbeddingModel.

    Designed to be lightweight — BM25 is pure Python, no external
    search engine needed.
    """

    def __init__(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: Optional[list[list[float]]],
        embedder,
    ):
        """
        Args:
            ids: ChromaDB document IDs.
            documents: Chunk text contents.
            metadatas: Chunk metadata (must align with documents).
            embeddings: Pre-computed embedding vectors.
            embedder: EmbeddingModel instance for query encoding.
        """
        self._ids = ids
        self._documents = documents
        self._metadatas = metadatas
        self._embeddings = embeddings
        self._embedder = embedder

        # Build BM25 index
        self._bm25 = self._build_bm25(documents) if documents else None

    def _build_bm25(self, documents: list[str]) -> Optional[BM25Okapi]:
        """Build BM25 index from tokenized documents."""
        if not documents:
            return None

        # Tokenize: character-level for Chinese, plus word-level for mixed text
        tokenized = [self._tokenize(doc) for doc in documents]
        return BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 indexing.

        Uses character-level tokenization (effective for Chinese)
        combined with space-splitting for any Latin words/numbers.
        """
        # Simple but effective for Chinese: split on non-alphanumeric,
        # keeping CJK characters as individual tokens
        tokens = []
        for char in text:
            if char.isspace():
                continue
            if '一' <= char <= '鿿' or '㐀' <= char <= '䶿':
                # CJK character — individual token
                tokens.append(char)
            elif char.isalnum():
                tokens.append(char.lower())
        return tokens

    def search(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
    ) -> list[dict]:
        """
        Hybrid search combining BM25 + vector with RRF fusion.

        Returns results in the standard format:
        {"content": str, "filename": str, "doc_id": str, "score": float, "source": str}
        """
        if not self._documents:
            return []

        # Run both retrievals
        bm25_results = self._run_bm25(query, top_k * 2)  # Oversample for fusion
        vector_results = self._run_vector(query, top_k * 2)

        # Fuse with RRF
        fused = self._rrf_fusion(bm25_results, vector_results, k=RRF_K)

        # Take top_k, deduplicating by doc_id
        seen_docs = set()
        final = []
        for item in fused[:top_k * 2]:
            doc_id = item.get("doc_id", "")
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            final.append(item)
            if len(final) >= top_k:
                break

        return final

    def search_bm25(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """Pure BM25 keyword search (no vector)."""
        return self._run_bm25(query, top_k)

    def search_vector(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """Pure vector semantic search (no BM25)."""
        return self._run_vector(query, top_k)

    def _run_bm25(self, query: str, top_k: int) -> list[dict]:
        """Execute BM25 search and return standardized results."""
        if not self._bm25 or not self._documents:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        if len(scores) == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            if idx >= len(self._ids):
                continue

            results.append({
                "id": self._ids[idx],
                "content": self._documents[idx] if idx < len(self._documents) else "",
                "filename": self._metadatas[idx].get("filename", "") if idx < len(self._metadatas) else "",
                "doc_id": self._metadatas[idx].get("doc_id", "") if idx < len(self._metadatas) else "",
                "score": round(float(scores[idx]), 4),
                "source": "bm25",
                "heading": self._metadatas[idx].get("heading", "") if idx < len(self._metadatas) else "",
            })

        return results

    def _run_vector(self, query: str, top_k: int) -> list[dict]:
        """Execute vector search and return standardized results."""
        if (
            self._embeddings is None
            or len(self._embeddings) == 0
            or len(self._documents) == 0
        ):
            return []

        try:
            query_vec = self._embedder.encode_query(query)
        except Exception as e:
            logger.error(f"Vector encoding failed for query: {e}")
            return []

        # Compute cosine similarity with all document embeddings
        doc_embeddings = np.array(self._embeddings)
        similarity = np.dot(doc_embeddings, query_vec)  # Normalized vectors

        top_indices = np.argsort(similarity)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarity[idx] <= 0:
                continue
            if idx >= len(self._ids):
                continue

            results.append({
                "id": self._ids[idx],
                "content": self._documents[idx] if idx < len(self._documents) else "",
                "filename": self._metadatas[idx].get("filename", "") if idx < len(self._metadatas) else "",
                "doc_id": self._metadatas[idx].get("doc_id", "") if idx < len(self._metadatas) else "",
                "score": round(float(similarity[idx]), 4),
                "source": "vector",
                "heading": self._metadatas[idx].get("heading", "") if idx < len(self._metadatas) else "",
            })

        return results

    def _rrf_fusion(
        self,
        results_a: list[dict],
        results_b: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion: combine two ranked lists into one.

        score(d) = sum( 1 / (k + rank_i(d)) ) across all rankings i

        Higher k = more smoothing, less emphasis on rank differences.
        Standard value k=60 works well for most scenarios.
        """
        if not results_a and not results_b:
            return []
        if not results_a:
            return results_b
        if not results_b:
            return results_a

        # Build score map from both ranked lists
        scores: dict[str, tuple[float, dict, list[str]]] = {}
        # id -> (total_rrf_score, best_result, sources)

        for rank, item in enumerate(results_a, 1):
            cid = item.get("id", "")
            rrf = 1.0 / (k + rank)
            if cid in scores:
                old_score, old_item, sources = scores[cid]
                scores[cid] = (old_score + rrf, old_item, sources + [item.get("source", "bm25")])
            else:
                scores[cid] = (rrf, dict(item), [item.get("source", "bm25")])

        for rank, item in enumerate(results_b, 1):
            cid = item.get("id", "")
            rrf = 1.0 / (k + rank)
            if cid in scores:
                old_score, old_item, sources = scores[cid]
                scores[cid] = (old_score + rrf, old_item, sources + [item.get("source", "vector")])
            else:
                scores[cid] = (rrf, dict(item), [item.get("source", "vector")])

        # Sort by RRF score descending
        fused = sorted(scores.values(), key=lambda x: x[0], reverse=True)

        # Build final results
        merged = []
        for rrf_score, item, sources in fused:
            item["score"] = round(rrf_score, 4)
            item["source"] = "+".join(sorted(set(sources)))
            merged.append(item)

        return merged

"""
Vector store — ChromaDB-backed persistent vector storage with hybrid search.

Replaces the old TF-IDF + cosine similarity implementation.

Architecture:
  Collection: "policy_docs"
  Each document = one PDF file
  Each chunk = one ChromaDB document with:
    - id: "{doc_id}_{chunk_id}"
    - embedding: BGE-small-zh-v1.5 vector
    - document: chunk text content
    - metadata: {doc_id, filename, chunk_id, chunk_index, heading, ...}

Supports three search modes:
  - "vector": pure semantic (embedding) search
  - "bm25": pure keyword search
  - "hybrid": RRF fusion of both (default)
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import RAG_TOP_K, RAG_SEARCH_MODE
from backend.rag.embedder import EmbeddingModel

logger = logging.getLogger(__name__)

# ChromaDB collection name — single collection for all policy documents
COLLECTION_NAME = "policy_docs"

# ChromaDB data directory (relative to project root)
import os
CHROMA_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "chroma_db"
)


class VectorStore:
    """
    ChromaDB-backed vector store for policy document chunks.

    Handles CRUD operations and semantic/hybrid search.
    Uses the global EmbeddingModel singleton for vectorization.
    """

    def __init__(self):
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None
        self._embedder: Optional[EmbeddingModel] = None
        self._bm25_index = None  # Lazy-loaded HybridSearcher
        self._init_client()

    def _init_client(self):
        """Initialize ChromaDB persistent client."""
        os.makedirs(CHROMA_DATA_DIR, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=CHROMA_DATA_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create the collection
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "Policy document chunks for RAG",
                "embedding_model": "bge-small-zh-v1.5",
            },
        )

        self._embedder = EmbeddingModel.get_instance()
        logger.info(
            f"VectorStore initialized. Collection '{COLLECTION_NAME}' "
            f"has {self._collection.count()} documents."
        )

    def add_document(
        self,
        filename: str,
        chunks: list[dict],
    ) -> str:
        """
        Add a document's chunks to the vector store (incremental).

        Each chunk is embedded and inserted into ChromaDB. No full re-indexing.

        Args:
            filename: Original filename (for metadata).
            chunks: List of chunk dicts from chunker.
                    {"id": str, "content": str, "metadata": dict}

        Returns:
            doc_id: Unique document identifier.
        """
        if not chunks:
            raise ValueError("Cannot add empty document — no chunks provided")

        doc_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        # Batch-encode all chunk texts for efficiency
        texts = [c["content"] for c in chunks]
        try:
            vectors = self._embedder.encode(texts, show_progress_bar=len(texts) > 10)
        except Exception as e:
            logger.error(f"Failed to encode chunks for '{filename}': {e}")
            raise RuntimeError(f"Embedding failed for '{filename}': {e}") from e

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{chunk['id']}"
            ids.append(chunk_id)
            documents.append(chunk["content"])
            metadatas.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_id": chunk["id"],
                "chunk_index": chunk.get("metadata", {}).get("chunk_index", i),
                "heading": chunk.get("metadata", {}).get("heading", ""),
                "char_count": len(chunk["content"]),
                "uploaded_at": now,
            })
            embeddings.append(vectors[i].tolist())

        # Insert into ChromaDB (incremental — no full re-index)
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        # Invalidate BM25 cache since documents changed
        self._bm25_index = None

        logger.info(
            f"Added document '{filename}' (id={doc_id}) "
            f"with {len(chunks)} chunks. "
            f"Collection size: {self._collection.count()}"
        )
        return doc_id

    def search(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        search_mode: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for the most relevant chunks.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            search_mode: "vector" | "bm25" | "hybrid".
                         Defaults to RAG_SEARCH_MODE config.

        Returns:
            List of result dicts:
            {"content": str, "filename": str, "doc_id": str, "score": float}
        """
        if self._collection.count() == 0:
            return []

        mode = search_mode or RAG_SEARCH_MODE

        if mode == "bm25":
            return self._bm25_search(query, top_k)
        elif mode == "hybrid":
            return self._hybrid_search(query, top_k)
        else:
            return self._vector_search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """Pure semantic (embedding) search via ChromaDB."""
        query_vec = self._embedder.encode_query(query).tolist()

        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results)

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """Pure BM25 keyword search."""
        searcher = self._get_hybrid_searcher()
        return searcher.search_bm25(query, top_k)

    def _hybrid_search(self, query: str, top_k: int) -> list[dict]:
        """Hybrid search: BM25 + vector with RRF fusion."""
        searcher = self._get_hybrid_searcher()
        return searcher.search(query, top_k)

    def _get_hybrid_searcher(self):
        """Lazy-load the HybridSearcher with current document data."""
        if self._bm25_index is None:
            from backend.rag.hybrid_search import HybridSearcher

            # Extract all documents from ChromaDB for BM25 indexing
            if self._collection.count() > 0:
                all_data = self._collection.get(
                    include=["documents", "metadatas", "embeddings"]
                )
                self._bm25_index = HybridSearcher(
                    ids=all_data["ids"],
                    documents=all_data["documents"],
                    metadatas=all_data["metadatas"],
                    embeddings=all_data["embeddings"],
                    embedder=self._embedder,
                )
            else:
                self._bm25_index = HybridSearcher(
                    ids=[], documents=[], metadatas=[], embeddings=[],
                    embedder=self._embedder,
                )

        return self._bm25_index

    def list_documents(self) -> list[dict]:
        """List all unique documents in the store."""
        if self._collection.count() == 0:
            return []

        all_data = self._collection.get(include=["metadatas"])
        metadatas = all_data["metadatas"]

        docs_map: dict[str, dict] = {}
        for meta in metadatas:
            doc_id = meta["doc_id"]
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta["filename"],
                    "uploaded_at": meta.get("uploaded_at", ""),
                    "chunk_count": 0,
                }
            docs_map[doc_id]["chunk_count"] += 1

        return list(docs_map.values())

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete all chunks belonging to a document.

        Uses ChromaDB's metadata filter for efficient deletion.
        """
        before = self._collection.count()

        # Get IDs of chunks to delete
        results = self._collection.get(
            where={"doc_id": doc_id},
            include=[],
        )

        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            self._bm25_index = None  # Invalidate BM25 cache
            logger.info(
                f"Deleted document {doc_id} ({len(results['ids'])} chunks). "
                f"Collection size: {self._collection.count()}"
            )
        else:
            logger.warning(f"Document {doc_id} not found for deletion")

        return self._collection.count() < before

    @property
    def count(self) -> int:
        """Total number of chunks in the store."""
        return self._collection.count()

    @property
    def collection(self) -> chromadb.Collection:
        """Direct access to ChromaDB collection (for advanced use)."""
        if self._collection is None:
            raise RuntimeError("VectorStore not initialized")
        return self._collection

    def _format_results(self, chroma_results: dict) -> list[dict]:
        """Convert ChromaDB query results to the standard result format."""
        formatted = []
        seen_docs = set()  # Dedup by doc_id, keep highest score per doc

        if not chroma_results["ids"] or not chroma_results["ids"][0]:
            return []

        ids = chroma_results["ids"][0]
        documents = chroma_results["documents"][0] if chroma_results["documents"] else []
        metadatas = chroma_results["metadatas"][0] if chroma_results["metadatas"] else []
        distances = chroma_results["distances"][0] if chroma_results["distances"] else []

        # Sort by distance ascending (lower = more similar for cosine)
        results = sorted(
            zip(ids, documents, metadatas, distances),
            key=lambda x: x[3],
        )

        for cid, content, meta, dist in results:
            doc_id = meta.get("doc_id", cid)
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)

            formatted.append({
                "content": content or "",
                "filename": meta.get("filename", ""),
                "doc_id": doc_id,
                "chunk_id": cid,
                "score": round(1.0 - dist, 4),  # Convert distance to similarity score
                "heading": meta.get("heading", ""),
            })

        return formatted[:RAG_TOP_K]

    def health_check(self) -> dict:
        """Check health of the vector store and embedding model."""
        try:
            count = self._collection.count()
            dim = self._embedder.dimension
            return {
                "status": "healthy",
                "chunk_count": count,
                "embedding_dimension": dim,
                "collection_name": COLLECTION_NAME,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# ============================================================
# Global singleton instance
# ============================================================
_store: Optional[VectorStore] = None


def _get_store() -> VectorStore:
    """Get or create the global VectorStore singleton."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def reset_store():
    """Reset the singleton (useful for testing)."""
    global _store
    _store = None


# ============================================================
# Public API — same signatures as the old vector_store.py
# ============================================================

def add_document(filename: str, chunks: list[dict]) -> str:
    """Add a document's chunks to the vector knowledge base."""
    return _get_store().add_document(filename, chunks)


def search(
    query: str,
    top_k: int = RAG_TOP_K,
    search_mode: Optional[str] = None,
) -> list[dict]:
    """Search the knowledge base for relevant chunks."""
    return _get_store().search(query, top_k, search_mode)


def list_documents() -> list[dict]:
    """List all unique documents in the knowledge base."""
    return _get_store().list_documents()


def delete_document(doc_id: str) -> bool:
    """Delete a document and all its chunks."""
    return _get_store().delete_document(doc_id)


def get_chunk_count() -> int:
    """Get total number of chunks in the knowledge base."""
    return _get_store().count


def health_check() -> dict:
    """Check vector store health."""
    return _get_store().health_check()

"""
RAG (Retrieval-Augmented Generation) module.

Enterprise-grade policy document understanding pipeline:
  1. PDF parsing with PyMuPDF (with OCR fallback)
  2. Semantic chunking for Chinese text
  3. BGE-small-zh-v1.5 embedding model
  4. ChromaDB persistent vector storage
  5. Hybrid search: BM25 + vector with RRF fusion
  6. Optional cross-encoder reranking

Key exports:
  - pdf_parser: extract_text, extract_metadata, needs_ocr
  - chunker: SemanticChunker, chunk_text
  - embedder: EmbeddingModel, encode_documents, encode_query
  - vector_store: add_document, search, list_documents, delete_document, health_check
  - hybrid_search: HybridSearcher
  - reranker: Reranker, rerank_if_enabled
"""

# ⚠️ HF_ENDPOINT must be set BEFORE any import that pulls in huggingface_hub
# (chromadb → huggingface_hub). Doing it here ensures it's the first thing
# that happens when the RAG package is loaded.
import os as _os
if _os.getenv("HF_ENDPOINT") is None:
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from backend.rag.pdf_parser import extract_text, extract_metadata, needs_ocr
from backend.rag.chunker import SemanticChunker, chunk_text
from backend.rag.embedder import EmbeddingModel, encode_documents, encode_query
from backend.rag.vector_store import (
    add_document,
    search,
    list_documents,
    delete_document,
    get_chunk_count,
    health_check,
)
from backend.rag.hybrid_search import HybridSearcher
from backend.rag.reranker import Reranker, rerank_if_enabled

__all__ = [
    # pdf_parser
    "extract_text",
    "extract_metadata",
    "needs_ocr",
    # chunker
    "SemanticChunker",
    "chunk_text",
    # embedder
    "EmbeddingModel",
    "encode_documents",
    "encode_query",
    # vector_store
    "add_document",
    "search",
    "list_documents",
    "delete_document",
    "get_chunk_count",
    "health_check",
    # hybrid_search
    "HybridSearcher",
    # reranker
    "Reranker",
    "rerank_if_enabled",
]

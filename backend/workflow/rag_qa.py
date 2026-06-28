"""
RAG QA Pipeline — retrieval → (rerank) → context building.

This is NOT a streaming workflow.  It's the retrieval half of RAG-based Q&A.
The generation (LLM streaming) half is handled by RAGAgent via the search_rag tool.

Provides:
  - retrieve_and_build_context(): synchronous pipeline used as a utility
  - build_rag_qa_workflow(): workflow definition for batch (non-streaming) use
"""

import json
import logging
from typing import Optional

from backend.config import RAG_TOP_K, RERANKER_ENABLED

logger = logging.getLogger(__name__)


def retrieve_and_build_context(
    query: str,
    top_k: int = RAG_TOP_K,
    rerank: bool | None = None,
) -> tuple[str, list[dict]]:
    """
    Core RAG retrieval pipeline: search → optional rerank → build context.

    Args:
        query: Search query.
        top_k: Number of results.
        rerank: Override RERANKER_ENABLED config (None = use config).

    Returns:
        (context_text, raw_results) — context_text is ready for prompt injection.
    """
    from backend.rag.vector_store import search

    should_rerank = rerank if rerank is not None else RERANKER_ENABLED

    # 1. Hybrid search
    results = search(query, top_k=top_k)

    if not results:
        return "", []

    # 2. Optional reranking
    if should_rerank and results:
        from backend.rag.reranker import rerank_if_enabled
        results = rerank_if_enabled(query, results, top_k)

    # 3. Build context string
    context_parts = []
    for i, r in enumerate(results):
        source_tag = f" (来源: {r.get('source', 'hybrid')})" if r.get("source") else ""
        context_parts.append(
            f"[文档{i+1}: {r['filename']}] "
            f"(相关度: {r['score']:.4f}){source_tag}\n"
            f"{r.get('heading', '')}\n"
            f"{r['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)
    return context, results


def build_rag_prompt(query: str, context: str) -> str:
    """Build the RAG-augmented prompt for LLM generation."""
    return f"""你是一个政策解读专家。请基于以下从官方政策文件中检索到的内容，回答用户的问题。

## 检索到的政策原文（按相关度排序）

{context}

## 回答要求

1. **严格基于上述原文**：只使用上面提供的政策内容来回答，不要编造
2. **标注来源**：引用时注明来自哪个文件
3. **说人话**：把政策语言翻译成普通人能听懂的大白话
4. **结构清晰**：使用分点、标题等方式组织回答
5. 如果检索内容不足以回答，请如实说明

## 用户问题

{query}

请回答："""


def build_rag_qa_prompt(query: str, top_k: int = RAG_TOP_K) -> str:
    """
    One-shot convenience: retrieve context and build the full RAG prompt.

    This is the primary entry point used by the RAG chat endpoint.
    Returns a prompt ready to send to Claude, or None if no results.
    """
    context, results = retrieve_and_build_context(query, top_k)
    if not context:
        return ""
    return build_rag_prompt(query, context)

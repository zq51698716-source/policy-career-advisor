"""
RAG retrieval tools — allow Agents to search the local knowledge base.

The ``search_rag`` tool is the bridge between the Agent layer and the RAG
pipeline.  When an agent needs to answer from uploaded policy documents it
calls this tool, which delegates to the vector store's hybrid search.
"""

import json
from typing import Any

from backend.config import RAG_TOP_K


# ============================================================
# Tool JSON Schema
# ============================================================
TOOL_SCHEMAS = {
    "search_rag": {
        "name": "search_rag",
        "description": (
            "本地政策知识库检索工具。在上传的政策PDF文件中搜索相关内容。"
            "适合回答需要引用具体政策文件原文的问题。返回最相关的文本片段及其来源文件。"
            "当用户明确提到'知识库'、'已上传的文件'、'政策文件'时优先使用此工具。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询，使用自然语言描述要查找的政策内容",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量，默认5，最多10",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# ============================================================
# Execution
# ============================================================
def _search_rag(query: str, top_k: int = 5) -> str:
    """Search the local RAG knowledge base."""
    from backend.rag.vector_store import search

    try:
        results = search(query, top_k=min(top_k, 10))

        if not results:
            return json.dumps({
                "query": query,
                "result_count": 0,
                "results": [],
                "note": "知识库中暂无相关文档，请建议用户上传相关政策PDF文件。",
            }, ensure_ascii=False, indent=2)

        # Optionally rerank
        from backend.config import RERANKER_ENABLED
        if RERANKER_ENABLED:
            from backend.rag.reranker import rerank_if_enabled
            results = rerank_if_enabled(query, results, min(top_k, 5))

        # Format for LLM consumption — keep it compact
        formatted = []
        for r in results:
            formatted.append({
                "source_file": r.get("filename", ""),
                "heading": r.get("heading", ""),
                "content": r.get("content", ""),
                "relevance": r.get("score", 0),
            })

        return json.dumps({
            "query": query,
            "result_count": len(formatted),
            "results": formatted,
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": f"知识库检索失败: {str(e)}"}, ensure_ascii=False)


EXECUTORS = {
    "search_rag": _search_rag,
}

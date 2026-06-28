"""
RAG routes — PDF upload, document management, RAG chat, standalone search.

Now uses:
  - Workflow for ingestion (parse → chunk → embed → store)
  - RAGAgent + rag_qa pipeline for RAG chat
  - Same management endpoints (unchanged)
"""

import os
import json
import re
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.rag.pdf_parser import extract_text
from backend.rag.chunker import SemanticChunker
from backend.rag.vector_store import (
    add_document,
    search,
    list_documents,
    delete_document,
    get_chunk_count,
    health_check,
)
from backend.rag.reranker import rerank_if_enabled
from backend.workflow.rag_ingestion import run_ingestion
from backend.workflow.rag_qa import retrieve_and_build_context, build_rag_prompt
from backend.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    RAG_TOP_K,
    MAX_UPLOAD_SIZE_MB,
    RERANKER_ENABLED,
)

router = APIRouter(prefix="/api/rag")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

_SAFE_FILENAME_RE = re.compile(r'[^\w一-鿿\-_.() （）]')


# === Request Models ===
class RAGChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=RAG_TOP_K, ge=1, le=20)


class RAGSearchRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=RAG_TOP_K, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(hybrid|vector|bm25)$")


# === PDF Upload (now using Workflow) ===
@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF policy file — parse, chunk, embed, and store (workflow-driven)."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            {"error": "仅支持 PDF 文件 (Only PDF files are supported)"},
            status_code=400,
        )

    safe_filename = _sanitize_filename(file.filename)
    if not safe_filename:
        return JSONResponse(
            {"error": "无效的文件名 (Invalid filename)"},
            status_code=400,
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            {
                "error": f"文件过大，最大支持 {MAX_UPLOAD_SIZE_MB}MB "
                         f"(File too large, max {MAX_UPLOAD_SIZE_MB}MB)"
            },
            status_code=413,
        )

    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        # Quick validation: can we extract text?
        text = extract_text(file_path)
        if not text.strip():
            return JSONResponse(
                {"error": "PDF 无法提取文字，可能是扫描件。请上传可搜索的 PDF。"},
                status_code=400,
            )

        # --- Run ingestion WORKFLOW ---
        ctx = await run_ingestion(file_path, safe_filename)

        return {
            "success": True,
            "doc_id": ctx.get("doc_id", ""),
            "filename": safe_filename,
            "chunk_count": ctx.get("chunk_count", 0),
            "text_length": ctx.get("text_length", 0),
            "message": f"成功解析 {ctx.get('chunk_count', 0)} 个文本块，已存入向量知识库",
        }

    except Exception as e:
        return JSONResponse(
            {"error": f"处理失败: {str(e)}"},
            status_code=500,
        )


# === Document Management (unchanged) ===
@router.get("/documents")
async def get_documents():
    """List all uploaded documents and their chunk counts."""
    docs = list_documents()
    total_chunks = get_chunk_count()
    return {
        "documents": docs,
        "total_chunks": total_chunks,
    }


@router.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    """Delete a document and all its chunks from the knowledge base."""
    ok = delete_document(doc_id)
    return {
        "success": ok,
        "message": "已删除" if ok else "文档不存在 (Document not found)",
    }


# === Standalone Search (unchanged) ===
@router.get("/search")
async def search_rag(
    q: str = Query(..., min_length=1, max_length=5000),
    top_k: int = Query(default=RAG_TOP_K, ge=1, le=20),
    mode: str = Query(default="hybrid", pattern="^(hybrid|vector|bm25)$"),
):
    """Standalone RAG search — returns retrieved chunks without LLM generation."""
    try:
        results = search(q, top_k=top_k, search_mode=mode)

        if RERANKER_ENABLED and results:
            results = rerank_if_enabled(q, results, top_k)

        return {
            "query": q,
            "mode": mode,
            "result_count": len(results),
            "results": results,
        }
    except Exception as e:
        return JSONResponse(
            {"error": f"检索失败 (Search failed): {str(e)}"},
            status_code=500,
        )


# === Health Check (unchanged) ===
@router.get("/health")
async def rag_health():
    """Check RAG system health: vector store + embedding model status."""
    return health_check()


# === RAG-Enhanced Chat (now using RAGAgent) ===
@router.post("/chat")
async def rag_chat(request: RAGChatRequest):
    """
    RAG-enhanced chat: retrieve relevant policy chunks, then stream Claude's answer.

    Now delegates to RAGAgent (which has the search_rag tool) for a unified
    agent experience.  The agent decides whether to search the KB, the web,
    or both.
    """
    return StreamingResponse(
        _rag_chat_stream(request.query, request.top_k),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _rag_chat_stream(query: str, top_k: int = 5) -> AsyncGenerator[str, None]:
    """
    RAG chat pipeline: retrieve → build context → stream Claude's answer.

    Uses the rag_qa pipeline for retrieval and context building,
    then streams Claude with the augmented prompt.
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        # 1. Retrieve and build context (via the rag_qa pipeline)
        yield _sse("thinking", "正在检索相关政策文件...")

        context, results = retrieve_and_build_context(query, top_k)

        if not context:
            yield _sse("content", "⚠️ 知识库中暂无相关文档。请先上传政策 PDF 文件。")
            yield _sse("done", "")
            return

        yield _sse("tool_result", json.dumps({
            "tool_name": "rag_search",
            "result_count": len(results),
            "summary": f"从 {len(set(r['filename'] for r in results))} 个文件中检索到 {len(results)} 个相关片段"
        }, ensure_ascii=False))

        # 2. Build RAG-augmented prompt
        rag_prompt = build_rag_prompt(query, context)

        # 3. Stream Claude's answer
        yield _sse("thinking", f"已检索到 {len(results)} 个相关片段，正在生成回答...")

        sq: asyncio.Queue[str] = asyncio.Queue(maxsize=1024)
        loop = asyncio.get_event_loop()

        def _stream_thread():
            def _put(msg):
                asyncio.run_coroutine_threadsafe(sq.put(msg), loop)

            try:
                with client.messages.stream(
                    model=CLAUDE_MODEL,
                    max_tokens=4096,
                    system="你是一个政策解读专家。请基于提供的政策原文回答问题。",
                    messages=[{"role": "user", "content": rag_prompt}]
                ) as stream:
                    for event in stream:
                        if event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "text_delta":
                                _put(_sse("content", delta.text))
                _put("__END__")
            except Exception as e:
                _put(_sse("error", str(e)))
                _put("__END__")

        loop.run_in_executor(None, _stream_thread)

        while True:
            msg = await sq.get()
            if msg == "__END__":
                break
            yield msg

        yield _sse("done", "")

    except Exception as e:
        yield _sse("error", str(e))


# === Helpers ===
def _sse(event_type: str, data) -> str:
    """Build an SSE-formatted string."""
    payload = {"type": event_type, "data": data}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and illegal characters."""
    name = os.path.basename(filename)
    name = name.replace("\x00", "")
    name = _SAFE_FILENAME_RE.sub("_", name)
    name = name.lstrip(".")
    if not name or not name.endswith(".pdf"):
        import uuid
        name = f"document_{uuid.uuid4().hex[:8]}.pdf"
    return name

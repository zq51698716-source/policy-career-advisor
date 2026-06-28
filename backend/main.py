"""
FastAPI application entry point.
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import validate_config, PORT, CLAUDE_MODEL
from backend.routers.chat import router as chat_router
from backend.routers.rag import router as rag_router

# Startup validation
try:
    validate_config()
except ValueError as e:
    print(f"\n{'='*60}")
    print(f"⚠️  配置错误: {e}")
    print(f"{'='*60}\n")

app = FastAPI(
    title="政策通 & 职业顾问 AI Agent",
    description="自动搜索最新政策、解读官方文件、提供职业建议的智能助手",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(chat_router)
app.include_router(rag_router)


# --- Health Check ---
@app.get("/api/health")
async def health():
    from backend.config import ANTHROPIC_API_KEY, TAVILY_API_KEY
    has_rag = False
    try:
        from backend.rag.vector_store import get_chunk_count
        has_rag = get_chunk_count() > 0
    except Exception:
        pass

    return {
        "status": "ok",
        "version": "2.0.0",
        "model": CLAUDE_MODEL,
        "claude_configured": bool(ANTHROPIC_API_KEY),
        "tavily_configured": bool(TAVILY_API_KEY),
        "rag_documents": has_rag,
    }


# --- Mount frontend static files (must be last) ---
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


# --- Entry point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
    )

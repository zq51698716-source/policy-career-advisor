"""
Application configuration — loads env vars for API keys, model settings, and RAG params.

Now includes orchestrator and workflow settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Auto-detect HuggingFace mirror for users behind firewalls.
if os.getenv("HF_ENDPOINT") is None:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- API Keys ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- Claude Model ---
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# --- Server ---
PORT = int(os.getenv("PORT", "8000"))

# --- Agent Configuration ---
MAX_TOOL_CALLS = 5  # Max tool-calling rounds per agent invocation
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "2048"))

# --- Orchestrator Configuration ---
ROUTING_STRATEGY = os.getenv("ROUTING_STRATEGY", "keyword")  # "keyword" | "llm"

# --- RAG Configuration ---
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
RAG_SEARCH_MODE = os.getenv("RAG_SEARCH_MODE", "hybrid")  # hybrid | vector | bm25
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))


def validate_config():
    """Validate that required config values are set."""
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    if missing:
        raise ValueError(
            f"缺少必要的环境变量: {', '.join(missing)}\n"
            f"请复制 .env.example 为 .env 并填入你的 API Keys"
        )

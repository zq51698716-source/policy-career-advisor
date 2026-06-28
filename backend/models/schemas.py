"""
Pydantic request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., description="消息角色: user / assistant")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """Chat request for POST /api/chat."""
    query: str = Field(..., description="用户提问", min_length=1, max_length=5000)
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="历史对话记录",
    )


class AgentType(str, Enum):
    """Agent types for routing."""
    POLICY = "policy"
    CAREER = "career"
    RAG = "rag"
    GENERAL = "general"


class SSEEvent(BaseModel):
    """SSE streaming event."""
    type: str = Field(..., description="Event type: thinking / tool_call / content / done / error")
    data: Optional[str] = Field(default=None, description="Event data")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = ""
    model: str = ""

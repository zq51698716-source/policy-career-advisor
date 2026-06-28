"""
Chat router — POST /api/chat (SSE streaming)

Now routes through the Orchestrator, which dispatches to the appropriate
specialised Agent (PolicyAgent, CareerAgent, or RAGAgent).
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.models.schemas import ChatRequest
from backend.orchestrator import get_orchestrator

router = APIRouter(prefix="/api")


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Unified chat endpoint — SSE streaming.

    The Orchestrator analyses the query and routes to the best agent:
      - Policy questions → PolicyAgent (web search)
      - Career questions → CareerAgent (market data)
      - Knowledge-base questions → RAGAgent (local documents)

    Request:
        {
            "query": "深圳2026年人才补贴政策",
            "history": [{"role": "user", "content": "..."}, ...]
        }

    Response: text/event-stream (SSE)
        data: {"type": "thinking", "data": "..."}
        data: {"type": "content", "data": "根据最新政策..."}
        data: {"type": "done", "data": ""}
    """
    history_dicts = [
        {"role": msg.role, "content": msg.content}
        for msg in request.history
    ]

    orch = get_orchestrator()

    return StreamingResponse(
        orch.chat_stream(request.query, history_dicts),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

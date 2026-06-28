"""
RAGAgent — knowledge-base-first Q&A, with web search fallback.
"""

from backend.agents.base import BaseAgent
from backend.agents.prompts import RAG_AGENT_PROMPT


class RAGAgent(BaseAgent):
    """Knowledge-base Q&A agent.

    Tools: search_rag (primary), search_web (fallback), get_current_time
    """

    system_prompt = RAG_AGENT_PROMPT
    tools = ["search_rag", "search_web", "get_current_time"]

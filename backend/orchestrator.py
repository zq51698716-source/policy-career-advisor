"""
Orchestrator — intent routing and central dispatch.

Analyses the user's query and routes it to the most appropriate agent:
  - Policy keywords → PolicyAgent (web search for policies)
  - Career keywords → CareerAgent (job market analysis)
  - Knowledge-base context → RAGAgent (local policy documents)

Can be extended with LLM-based routing for ambiguous queries.
"""

import re
import logging
from typing import AsyncGenerator

from backend.agents.base import BaseAgent
from backend.agents.policy_agent import PolicyAgent
from backend.agents.career_agent import CareerAgent
from backend.agents.rag_agent import RAGAgent
from backend.agents.prompts import GENERAL_AGENT_PROMPT

logger = logging.getLogger(__name__)


# ============================================================
# Keyword routing tables
# ============================================================

POLICY_KEYWORDS = [
    "政策", "补贴", "社保", "人才引进", "住房补贴", "落户",
    "创业补贴", "就业扶持", "税收优惠", "公积金", "政府",
    "申请条件", "申请流程", "官方发布", "gov", "补贴标准",
    "人才安居", "保障房", "公租房", "人才房", "安居房",
    "引进", "优惠", "减免", "扶持", "资助", "专项资金",
]

CAREER_KEYWORDS = [
    "薪资", "工资", "薪水", "待遇", "薪酬", "收入",
    "岗位", "招聘", "求职", "面试", "简历", "cv",
    "行业趋势", "发展前景", "技能要求", "职位",
    "跳槽", "转行", "职业规划", "职业发展",
    "前端", "后端", "工程师", "产品经理", "数据分析",
    "AI", "人工智能", "机器学习", "开发", "测试",
]

RAG_KEYWORDS = [
    "知识库", "上传的文件", "政策文件", "本地文件",
    "已上传", "pdf", "文件里", "文档中",
]


def _has_rag_documents() -> bool:
    """Check if the knowledge base has any documents."""
    try:
        from backend.rag.vector_store import get_chunk_count
        return get_chunk_count() > 0
    except Exception:
        return False


def _match_keywords(query: str, keywords: list[str]) -> int:
    """Count how many keywords from the list appear in the query."""
    q_lower = query.lower()
    return sum(1 for kw in keywords if kw.lower() in q_lower)


# ================================================================
# Orchestrator
# ================================================================
class Orchestrator:
    """
    Intent-aware routing dispatcher.

    Usage:
        orch = Orchestrator()
        async for sse in orch.chat_stream("深圳人才补贴政策", history=[]):
            ...
    """

    def __init__(self):
        self._policy_agent: PolicyAgent | None = None
        self._career_agent: CareerAgent | None = None
        self._rag_agent: RAGAgent | None = None

    @property
    def policy_agent(self) -> PolicyAgent:
        if self._policy_agent is None:
            self._policy_agent = PolicyAgent()
        return self._policy_agent

    @property
    def career_agent(self) -> CareerAgent:
        if self._career_agent is None:
            self._career_agent = CareerAgent()
        return self._career_agent

    @property
    def rag_agent(self) -> RAGAgent:
        if self._rag_agent is None:
            self._rag_agent = RAGAgent()
        return self._rag_agent

    # ============================================================
    # Routing
    # ============================================================
    def route(self, query: str) -> BaseAgent:
        """
        Determine the best agent for this query.

        Priority:
          1. RAG keywords + knowledge base has documents → RAGAgent
          2. Policy keywords dominate → PolicyAgent
          3. Career keywords dominate → CareerAgent
          4. RAG keywords (even if KB empty — agent will guide user) → RAGAgent
          5. Default → PolicyAgent
        """
        policy_score = _match_keywords(query, POLICY_KEYWORDS)
        career_score = _match_keywords(query, CAREER_KEYWORDS)
        rag_score = _match_keywords(query, RAG_KEYWORDS)

        # RAG with documents available is highest priority
        if rag_score > 0 and _has_rag_documents():
            logger.info(f"Routing to RAGAgent (rag={rag_score}) — query: {query[:60]}")
            return self.rag_agent

        # Career strongly matched
        if career_score > policy_score and career_score >= 2:
            logger.info(f"Routing to CareerAgent (career={career_score}) — query: {query[:60]}")
            return self.career_agent

        # Policy matched
        if policy_score > 0:
            logger.info(f"Routing to PolicyAgent (policy={policy_score}) — query: {query[:60]}")
            return self.policy_agent

        # Career weakly matched
        if career_score > 0:
            logger.info(f"Routing to CareerAgent (career={career_score}) — query: {query[:60]}")
            return self.career_agent

        # RAG keywords but no documents — still route to RAGAgent so it can
        # guide the user to upload documents
        if rag_score > 0:
            logger.info(f"Routing to RAGAgent (rag={rag_score}, no docs) — query: {query[:60]}")
            return self.rag_agent

        # Default
        logger.info(f"Routing to PolicyAgent (default) — query: {query[:60]}")
        return self.policy_agent

    # ============================================================
    # Unified chat stream
    # ============================================================
    async def chat_stream(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Route to the best agent and stream its response.
        """
        agent = self.route(query)

        # If routing to general PolicyAgent but the query doesn't match
        # policy keywords, use the general prompt for broader coverage
        if isinstance(agent, PolicyAgent) and _match_keywords(query, POLICY_KEYWORDS) == 0:
            # Save original, use general prompt for this turn
            original = agent.system_prompt
            agent.system_prompt = GENERAL_AGENT_PROMPT
            try:
                async for event in agent.chat_stream(query, history):
                    yield event
            finally:
                agent.system_prompt = original
        else:
            async for event in agent.chat_stream(query, history):
                yield event


# Global singleton
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    """Get or create the global Orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator

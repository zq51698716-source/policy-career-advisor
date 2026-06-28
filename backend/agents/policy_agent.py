"""
PolicyAgent — web-search-driven policy lookup and interpretation.
"""

from backend.agents.base import BaseAgent
from backend.agents.prompts import POLICY_AGENT_PROMPT


class PolicyAgent(BaseAgent):
    """Policy search & interpretation agent.

    Tools: search_web, search_policy, get_current_time
    """

    system_prompt = POLICY_AGENT_PROMPT
    tools = ["search_web", "search_policy", "get_current_time"]

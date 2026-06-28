"""
CareerAgent — job market analysis, resume review, interview coaching.
"""

from backend.agents.base import BaseAgent
from backend.agents.prompts import CAREER_AGENT_PROMPT


class CareerAgent(BaseAgent):
    """Career & job market analysis agent.

    Tools: search_job_market, search_web, get_current_time
    """

    system_prompt = CAREER_AGENT_PROMPT
    tools = ["search_job_market", "search_web", "get_current_time"]

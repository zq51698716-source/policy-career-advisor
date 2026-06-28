"""
Workflow step types — the building blocks of a workflow DAG.

Each step has:
  - name: unique identifier within the workflow
  - depends_on: list of step names that must complete first
  - on_error: "fail" | "skip" | "continue"

Concrete types:
  - FunctionStep: calls a plain Python function
  - ToolStep: calls a registered tool by name
  - AgentStep: invokes an agent (LLM + tools)
  - ParallelStep: runs multiple sub-steps concurrently
"""

from typing import Any, Callable, Awaitable
from dataclasses import dataclass, field


@dataclass
class StepResult:
    """Output of a completed step."""
    step_name: str
    status: str          # "ok" | "skipped" | "failed"
    output: Any = None
    error: str | None = None


@dataclass
class WorkflowStep:
    """Base step definition."""
    name: str
    depends_on: list[str] = field(default_factory=list)
    on_error: str = "fail"  # "fail" | "skip" | "continue"


@dataclass
class FunctionStep(WorkflowStep):
    """Calls a regular (sync or async) Python function."""
    fn: Callable | None = None
    kwargs: dict = field(default_factory=dict)


@dataclass
class ToolStep(WorkflowStep):
    """Calls a registered tool by name."""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)


@dataclass
class AgentStep(WorkflowStep):
    """Invokes an agent with a prompt and returns the result."""
    agent: Any = None                    # BaseAgent instance
    prompt: str = ""
    system_prompt_override: str | None = None


@dataclass
class ParallelStep(WorkflowStep):
    """Runs multiple sub-steps concurrently."""
    steps: list[WorkflowStep] = field(default_factory=list)

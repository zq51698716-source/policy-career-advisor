"""
WorkflowEngine — lightweight DAG executor for multi-step pipelines.

Executes steps in topological order, respecting dependencies.
Supports parallel execution of steps whose dependencies are met.
Each step can be a FunctionStep, ToolStep, AgentStep, or ParallelStep.
"""

import asyncio
import logging
from typing import Any

from backend.workflow.steps import (
    WorkflowStep, FunctionStep, ToolStep, AgentStep, ParallelStep, StepResult,
)
from backend.tools import execute_tool

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Lightweight DAG workflow executor."""

    def __init__(self, name: str = "workflow"):
        self.name = name

    async def run(self, steps: list[WorkflowStep]) -> dict[str, StepResult]:
        """
        Execute steps respecting their dependency DAG.

        Returns a dict mapping step name → StepResult.
        """
        results: dict[str, StepResult] = {}
        remaining = list(steps)
        pending_futures: dict[str, asyncio.Task] = {}

        while remaining or pending_futures:
            # Find steps whose dependencies are all resolved
            ready = []
            still_waiting = []
            for step in remaining:
                if self._dependencies_met(step, results):
                    ready.append(step)
                else:
                    still_waiting.append(step)
            remaining = still_waiting

            if not ready and not pending_futures:
                if remaining:
                    unresolved = [s.name for s in remaining]
                    logger.warning(f"Workflow '{self.name}': unresolved steps (circular?): {unresolved}")
                break

            # Execute ready steps in parallel
            tasks = {}
            for step in ready:
                task = asyncio.create_task(self._execute_step(step, results))
                tasks[task] = step.name
                pending_futures[step.name] = task

            if not tasks:
                # Nothing ready, wait for pending futures to complete
                if pending_futures:
                    done, _ = await asyncio.wait(
                        pending_futures.values(),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        name = tasks.get(task, "unknown")
                        try:
                            result = task.result()
                        except Exception as e:
                            result = StepResult(name, "failed", error=str(e))
                        results[name] = result
                        pending_futures.pop(name, None)
                continue

            # Wait for all ready tasks to complete
            for task in asyncio.as_completed(list(tasks)):
                name = tasks[task]
                try:
                    result = await task
                except Exception as e:
                    result = StepResult(name, "failed", error=str(e))
                results[name] = result
                pending_futures.pop(name, None)

        return results

    def _dependencies_met(self, step: WorkflowStep, results: dict[str, StepResult]) -> bool:
        """Check if all dependencies have finished (ok, skipped, or failed-but-continue)."""
        for dep in step.depends_on:
            if dep not in results:
                return False
            r = results[dep]
            if r.status == "failed":
                # Only block if the dependent step was "fail" on error
                return False
        return True

    async def _execute_step(self, step: WorkflowStep, results: dict[str, StepResult]) -> StepResult:
        """Execute a single step, handling errors per on_error policy."""
        try:
            if isinstance(step, FunctionStep):
                output = await self._run_function(step)
            elif isinstance(step, ToolStep):
                output = await self._run_tool(step)
            elif isinstance(step, AgentStep):
                output = await self._run_agent(step)
            elif isinstance(step, ParallelStep):
                output = await self._run_parallel(step)
            else:
                raise ValueError(f"Unknown step type: {type(step).__name__}")

            return StepResult(step.name, "ok", output=output)

        except Exception as e:
            logger.error(f"Step '{step.name}' failed: {e}")
            if step.on_error == "skip":
                return StepResult(step.name, "skipped", error=str(e))
            elif step.on_error == "continue":
                return StepResult(step.name, "failed", error=str(e))
            else:  # "fail"
                raise

    # --- Step runners ---

    async def _run_function(self, step: FunctionStep) -> Any:
        if step.fn is None:
            raise ValueError(f"FunctionStep '{step.name}' has no fn")
        result = step.fn(**step.kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def _run_tool(self, step: ToolStep) -> str:
        return await asyncio.to_thread(execute_tool, step.tool_name, step.tool_input)

    async def _run_agent(self, step: AgentStep) -> str:
        """Run an agent and collect its full text output."""
        if step.agent is None:
            raise ValueError(f"AgentStep '{step.name}' has no agent")

        # Temporarily override system prompt if requested
        original_prompt = step.agent.system_prompt
        if step.system_prompt_override:
            step.agent.system_prompt = step.system_prompt_override

        try:
            parts = []
            async for sse_msg in step.agent.chat_stream(step.prompt):
                # Collect content events into a single string
                parsed = step.agent._try_parse_sse(sse_msg)
                if parsed and parsed["type"] == "content":
                    parts.append(parsed["data"])
            return "".join(parts)
        finally:
            step.agent.system_prompt = original_prompt

    async def _run_parallel(self, step: ParallelStep) -> list[StepResult]:
        sub_results = await asyncio.gather(
            *[self._execute_step(s, {}) for s in step.steps],
            return_exceptions=True,
        )
        output = []
        for s, r in zip(step.steps, sub_results):
            if isinstance(r, Exception):
                output.append(StepResult(s.name, "failed", error=str(r)))
            else:
                output.append(r)
        return output

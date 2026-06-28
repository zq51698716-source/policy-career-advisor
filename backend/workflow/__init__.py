"""
Workflow package — DAG-based multi-step pipeline orchestration.

Exports:
  - WorkflowEngine: lightweight DAG executor
  - Steps: FunctionStep, ToolStep, AgentStep, ParallelStep
  - RAG ingestion workflow builder
  - RAG QA pipeline utilities
"""

from backend.workflow.engine import WorkflowEngine
from backend.workflow.steps import (
    WorkflowStep as Step,
    FunctionStep,
    ToolStep,
    AgentStep,
    ParallelStep,
    StepResult,
)
from backend.workflow.rag_ingestion import build_ingestion_workflow, run_ingestion
from backend.workflow.rag_qa import (
    retrieve_and_build_context,
    build_rag_prompt,
    build_rag_qa_prompt,
)

__all__ = [
    "WorkflowEngine",
    "Step",
    "FunctionStep",
    "ToolStep",
    "AgentStep",
    "ParallelStep",
    "StepResult",
    "build_ingestion_workflow",
    "run_ingestion",
    "retrieve_and_build_context",
    "build_rag_prompt",
    "build_rag_qa_prompt",
]

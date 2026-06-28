"""
RAG Ingestion Workflow: Parse → Chunk → Embed → Store

Orchestrates the full pipeline from a PDF file path to vector store insertion.
Used by the /api/rag/upload endpoint.
"""

import logging
from backend.workflow.steps import FunctionStep
from backend.workflow.engine import WorkflowEngine

logger = logging.getLogger(__name__)


def build_ingestion_workflow(file_path: str, filename: str) -> list[FunctionStep]:
    """
    Build a RAG ingestion workflow as a list of FunctionSteps.

    The steps form a linear DAG:
      parse → chunk → embed_and_store

    Each step passes its output to the next via a shared context dict.
    """
    ctx: dict = {"file_path": file_path, "filename": filename}

    # --- Step 1: Parse PDF ---
    def parse_pdf(ctx: dict) -> dict:
        from backend.rag.pdf_parser import extract_text
        text = extract_text(ctx["file_path"])
        if not text.strip():
            raise ValueError("PDF 无法提取文字，可能是扫描件")
        ctx["text"] = text
        ctx["text_length"] = len(text)
        return ctx

    parse_step = FunctionStep(
        name="parse_pdf",
        fn=parse_pdf,
        kwargs={"ctx": ctx},
        on_error="fail",
    )

    # --- Step 2: Semantic Chunking ---
    def chunk_text(ctx: dict) -> dict:
        from backend.rag.chunker import SemanticChunker
        chunker = SemanticChunker()
        chunks = chunker.chunk(ctx["text"], filename=ctx["filename"])
        if not chunks:
            raise ValueError("PDF 内容为空")
        ctx["chunks"] = chunks
        ctx["chunk_count"] = len(chunks)
        return ctx

    chunk_step = FunctionStep(
        name="chunk_text",
        fn=chunk_text,
        kwargs={"ctx": ctx},
        depends_on=["parse_pdf"],
        on_error="fail",
    )

    # --- Step 3: Embed + Store ---
    def embed_and_store(ctx: dict) -> dict:
        from backend.rag.vector_store import add_document
        doc_id = add_document(ctx["filename"], ctx["chunks"])
        ctx["doc_id"] = doc_id
        return ctx

    store_step = FunctionStep(
        name="embed_and_store",
        fn=embed_and_store,
        kwargs={"ctx": ctx},
        depends_on=["chunk_text"],
        on_error="fail",
    )

    return [parse_step, chunk_step, store_step]


async def run_ingestion(file_path: str, filename: str) -> dict:
    """
    Run the full RAG ingestion workflow.

    Returns the workflow context dict with keys:
      text, text_length, chunks, chunk_count, doc_id
    """
    steps = build_ingestion_workflow(file_path, filename)
    engine = WorkflowEngine(name="rag_ingestion")
    results = await engine.run(steps)

    # Extract the final context from the last step
    for step_name, result in results.items():
        if result.status == "failed":
            raise RuntimeError(f"Ingestion step '{step_name}' failed: {result.error}")

    # Return the context from the store step
    store_result = results.get("embed_and_store")
    if store_result and store_result.output:
        return store_result.output
    return {}

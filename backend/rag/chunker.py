"""
Semantic text chunker — splits documents into semantically coherent chunks
using RecursiveCharacterTextSplitter with Chinese-aware separators.

Replaces the old fixed-size chunking in pdf_parser.py.
"""

import re
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP


# Chinese-optimized separators: policy documents are often structured with
# numbered clauses, section headings, and paragraph breaks.
CHINESE_SEPARATORS = [
    # Major structural breaks
    "\n\n\n",
    "\n\n",
    "\n",
    # Chinese punctuation boundaries
    "。",
    "；",
    "：",
    # Numbered clauses common in policy documents
    "第",
    # Whitespace
    " ",
    "",
]

# Patterns that indicate a heading / section boundary in Chinese policy docs
HEADING_PATTERNS = re.compile(
    r"^[  \t]*"
    r"("
    r"(?:第[一二三四五六七八九十百千\d]+[章节条]|"   # 第X章/第X条
    r"[一二三四五六七八九十]+、|"                     # 一、二、三、
    r"\(\d+\)|"                                       # (1) (2)
    r"\d+[\.\、]|"                                    # 1. 2.
    r"[（(]\d+[）)])"                                 # (1) (2)
    r")"
)


class SemanticChunker:
    """
    Splits text into semantically coherent chunks for Chinese policy documents.

    Uses RecursiveCharacterTextSplitter with separators tuned for Chinese text,
    then enriches each chunk with metadata about its position in the document.
    """

    def __init__(
        self,
        chunk_size: int = RAG_CHUNK_SIZE,
        chunk_overlap: int = RAG_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            separators=CHINESE_SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            keep_separator="end",  # Keep separator at end of chunk for readability
        )

    def chunk(
        self,
        text: str,
        filename: Optional[str] = None,
        page_offset: int = 0,
    ) -> list[dict]:
        """
        Split text into chunks with metadata.

        Args:
            text: Full document text.
            filename: Document filename for metadata.
            page_offset: Starting page number (for multi-file scenarios).

        Returns:
            List of chunk dicts:
            {"id": str, "content": str, "metadata": {"chunk_index": int, ...}}
        """
        if not text or not text.strip():
            return []

        # Pre-process: detect and mark section boundaries
        # This helps the splitter respect policy document structure
        text = self._mark_headings(text)

        # Split using RecursiveCharacterTextSplitter
        raw_chunks = self._splitter.split_text(text)

        # Build structured chunk dicts
        chunks = []
        for idx, content in enumerate(raw_chunks):
            content = content.strip()
            if not content:
                continue

            # Detect if this chunk starts with a heading
            heading = self._detect_heading(content)

            chunks.append({
                "id": f"chunk_{idx}",
                "content": content,
                "metadata": {
                    "chunk_index": idx,
                    "total_chunks": len(raw_chunks),
                    "filename": filename or "",
                    "heading": heading or "",
                    "char_count": len(content),
                },
            })

        return chunks

    def _mark_headings(self, text: str) -> str:
        """
        Insert a marker before detected headings so the splitter
        respects document structure boundaries.
        """
        lines = text.split("\n")
        marked = []
        for line in lines:
            stripped = line.strip()
            if HEADING_PATTERNS.match(stripped):
                # Add extra newline before headings to create split boundary
                marked.append("")
                marked.append(line)
            else:
                marked.append(line)
        return "\n".join(marked)

    def _detect_heading(self, text: str) -> Optional[str]:
        """Extract the first heading from a chunk, if any."""
        first_line = text.strip().split("\n")[0].strip()
        if HEADING_PATTERNS.match(first_line) and len(first_line) < 80:
            return first_line
        return None


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200,
) -> list[dict]:
    """
    Convenience function matching the signature of the old pdf_parser.chunk_text().

    Kept for backward compatibility.
    """
    chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    return chunker.chunk(text)

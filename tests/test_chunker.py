"""
Unit tests for chunker — semantic text chunking for Chinese policy documents.
"""

import pytest
from backend.rag.chunker import SemanticChunker, chunk_text

# Sample Chinese policy text with various structural elements
SAMPLE_POLICY_TEXT = """深圳市人才引进实施办法

第一章 总则

第一条 为深入推进人才强市战略，优化人才发展环境，根据《关于促进人才优先发展的若干措施》，制定本办法。

第二条 本办法适用于深圳市行政区域内的人才引进及相关服务管理工作。

第三条 人才引进工作应当遵循以下原则：
（一）服务发展原则。围绕我市经济社会发展需要，精准引进各类人才。
（二）市场导向原则。充分发挥市场在人才资源配置中的决定性作用。
（三）分类施策原则。根据不同人才类型和层次，制定差异化政策措施。

第二章 引进对象与条件

第四条 人才引进分为以下类别：
（一）高层次人才引进；
（二）留学回国人员引进；
（三）应届毕业生接收；
（四）在职人才引进。

第五条 高层次人才应当符合以下条件之一：
1. 中国科学院院士、中国工程院院士；
2. 国家最高科学技术奖获得者；
3. 国家自然科学奖、技术发明奖、科技进步奖二等奖以上的主要完成人。

第三章 支持政策

第六条 对新引进的杰出人才，给予以下支持：
（一）科研经费资助。最高不超过1000万元。
（二）住房补贴。提供免租住房一套，或给予相应租房补贴。
（三）生活补贴。每月发放生活补贴。

第七条 人才安居政策包括：
1. 人才住房。提供面向人才配租配售的住房。
2. 租房补贴。对自行租房的人才给予货币补贴。
3. 购房补贴。对在深购买首套住房的人才给予一次性补贴。

第四章 申请程序

第八条 人才引进按以下程序办理：
（一）用人单位申报；
（二）主管部门审核；
（三）公示；
（四）办理相关手续。

第九条 申请人应当提交以下材料：
1. 身份证明文件；
2. 学历学位证书；
3. 工作经历证明；
4. 其他相关证明材料。
"""

SHORT_TEXT = "深圳市近日发布了新的人才补贴政策，对符合条件的高层次人才提供住房补贴和生活补贴。"


class TestSemanticChunker:
    """Tests for SemanticChunker class."""

    def test_chunk_returns_list(self):
        """Should return a list of chunk dicts."""
        chunker = SemanticChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk(SAMPLE_POLICY_TEXT, filename="test.pdf")
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_has_required_keys(self):
        """Each chunk should have id, content, metadata."""
        chunker = SemanticChunker()
        chunks = chunker.chunk(SAMPLE_POLICY_TEXT, filename="test.pdf")
        for c in chunks:
            assert "id" in c
            assert "content" in c
            assert "metadata" in c
            assert "chunk_index" in c["metadata"]
            assert "filename" in c["metadata"]

    def test_chunk_content_not_empty(self):
        """No chunk should have empty content."""
        chunker = SemanticChunker()
        chunks = chunker.chunk(SAMPLE_POLICY_TEXT)
        for c in chunks:
            assert c["content"].strip(), f"Chunk {c['id']} has empty content"

    def test_chunk_respects_size_limit(self):
        """Chunks should not grossly exceed the configured chunk_size."""
        chunker = SemanticChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk(SAMPLE_POLICY_TEXT)
        for c in chunks:
            # Allow some flexibility (RecursiveCharacterTextSplitter is approximate)
            assert len(c["content"]) <= 600, (
                f"Chunk {c['id']} is {len(c['content'])} chars, expected <= 600"
            )

    def test_chunk_preserves_heading(self):
        """Chunks that start with a heading should have it in metadata."""
        chunker = SemanticChunker(chunk_size=400, chunk_overlap=50)
        chunks = chunker.chunk(SAMPLE_POLICY_TEXT)
        # At least one chunk should have a heading detected
        headings = [c["metadata"].get("heading") for c in chunks if c["metadata"].get("heading")]
        assert len(headings) > 0, "Some chunks should have detected headings"

    def test_chunk_total_chunks_metadata(self):
        """Metadata should include total_chunks count."""
        chunker = SemanticChunker()
        chunks = chunker.chunk(SAMPLE_POLICY_TEXT)
        total = chunks[0]["metadata"]["total_chunks"]
        assert total == len(chunks)

    def test_chunk_empty_input(self):
        """Should return empty list for empty text."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("")
        assert chunks == []

    def test_chunk_whitespace_only(self):
        """Should return empty list for whitespace-only text."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("   \n\n   ")
        assert chunks == []

    def test_chunk_short_text(self):
        """Short text should produce a single chunk."""
        chunker = SemanticChunker(chunk_size=800)
        chunks = chunker.chunk(SHORT_TEXT, filename="short.pdf")
        assert len(chunks) == 1
        assert SHORT_TEXT.strip() in chunks[0]["content"]

    def test_chunk_filename_in_metadata(self):
        """Filename should be propagated to chunk metadata."""
        chunker = SemanticChunker()
        chunks = chunker.chunk(SAMPLE_POLICY_TEXT, filename="深圳人才政策2026.pdf")
        for c in chunks:
            assert c["metadata"]["filename"] == "深圳人才政策2026.pdf"

    def test_chunk_char_count_in_metadata(self):
        """Metadata should include char_count."""
        chunker = SemanticChunker()
        chunks = chunker.chunk(SAMPLE_POLICY_TEXT)
        for c in chunks:
            assert c["metadata"]["char_count"] == len(c["content"])

    def test_custom_chunk_parameters(self):
        """Should respect custom chunk_size and chunk_overlap."""
        chunker = SemanticChunker(chunk_size=200, chunk_overlap=30)
        chunks = chunker.chunk(SAMPLE_POLICY_TEXT)
        # Smaller chunk_size should produce more chunks
        assert len(chunks) > 1


class TestChunkTextFunction:
    """Tests for the convenience function chunk_text."""

    def test_chunk_text_returns_list(self):
        """Should return list of chunk dicts."""
        chunks = chunk_text(SAMPLE_POLICY_TEXT, chunk_size=300, overlap=50)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_text_default_parameters(self):
        """Should work with default chunk_size and overlap."""
        chunks = chunk_text(SAMPLE_POLICY_TEXT)
        assert len(chunks) > 0

    def test_chunk_text_empty_input(self):
        """Should return empty list for empty input."""
        chunks = chunk_text("")
        assert chunks == []

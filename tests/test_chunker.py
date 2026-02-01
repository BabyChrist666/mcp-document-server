"""Tests for document chunking."""

import pytest
from mcp_doc_server.chunker import chunk_text, chunk_by_pages


class TestChunkText:
    def test_basic_chunking(self):
        text = "Hello world. This is a test document. It has multiple sentences."
        chunks = chunk_text(text, doc_id="test", chunk_size=30, overlap=5)
        assert len(chunks) > 0
        assert all(c.doc_id == "test" for c in chunks)

    def test_empty_text_returns_empty(self):
        chunks = chunk_text("", doc_id="test")
        assert chunks == []

    def test_whitespace_only_returns_empty(self):
        chunks = chunk_text("   \n\n  ", doc_id="test")
        assert chunks == []

    def test_chunk_ids_are_unique(self):
        text = "A " * 500
        chunks = chunk_text(text, doc_id="test", chunk_size=100, overlap=10)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunks_are_indexed(self):
        text = "Word " * 200
        chunks = chunk_text(text, doc_id="test", chunk_size=100, overlap=10)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_metadata_has_word_count(self):
        text = "one two three four five six seven eight nine ten"
        chunks = chunk_text(text, doc_id="test", chunk_size=1000)
        assert chunks[0].metadata["word_count"] > 0

    def test_overlap_creates_more_chunks(self):
        text = "Word " * 100
        no_overlap = chunk_text(text, doc_id="t", chunk_size=100, overlap=0)
        with_overlap = chunk_text(text, doc_id="t", chunk_size=100, overlap=40)
        assert len(with_overlap) >= len(no_overlap)

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("test", doc_id="t", chunk_size=0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError):
            chunk_text("test", doc_id="t", overlap=-1)

    def test_overlap_gte_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("test", doc_id="t", chunk_size=10, overlap=10)

    def test_start_end_char_positions(self):
        text = "ABCDE " * 50
        chunks = chunk_text(text, doc_id="t", chunk_size=50, overlap=5)
        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char


class TestChunkByPages:
    def test_basic_page_chunking(self):
        pages = ["Page one content here.", "Page two content here."]
        chunks = chunk_by_pages(pages, doc_id="doc1", chunk_size=100)
        assert len(chunks) > 0

    def test_page_metadata_present(self):
        pages = ["First page text.", "Second page text."]
        chunks = chunk_by_pages(pages, doc_id="doc1", chunk_size=1000)
        for c in chunks:
            assert "page" in c.metadata
            assert c.metadata["page"] >= 1

    def test_empty_pages_skipped(self):
        pages = ["Content.", "", "  ", "More content."]
        chunks = chunk_by_pages(pages, doc_id="doc1", chunk_size=1000)
        page_nums = {c.metadata["page"] for c in chunks}
        assert 2 not in page_nums
        assert 3 not in page_nums

    def test_all_chunks_share_doc_id(self):
        pages = ["Page A.", "Page B."]
        chunks = chunk_by_pages(pages, doc_id="mydoc", chunk_size=1000)
        for c in chunks:
            assert c.doc_id == "mydoc"

    def test_reindexing(self):
        pages = ["Page one.", "Page two.", "Page three."]
        chunks = chunk_by_pages(pages, doc_id="d", chunk_size=1000)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

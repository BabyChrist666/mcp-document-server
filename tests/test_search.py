"""Tests for the search module (cosine similarity, index management)."""

import pytest
from mcp_doc_server.search import DocumentIndex, SearchResult
from mcp_doc_server.chunker import Chunk


class TestCosineSimilarity:
    def test_identical_vectors(self):
        score = DocumentIndex._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(score - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        score = DocumentIndex._cosine_similarity([1, 0, 0], [0, 1, 0])
        assert abs(score) < 1e-6

    def test_opposite_vectors(self):
        score = DocumentIndex._cosine_similarity([1, 0], [-1, 0])
        assert abs(score - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        score = DocumentIndex._cosine_similarity([0, 0, 0], [1, 2, 3])
        assert score == 0.0

    def test_similar_vectors_high_score(self):
        score = DocumentIndex._cosine_similarity([1, 2, 3], [1, 2, 4])
        assert score > 0.9


class TestDocumentIndex:
    def test_init_defaults(self):
        idx = DocumentIndex(cohere_api_key="test")
        assert idx.total_chunks == 0

    def test_list_documents_empty(self):
        idx = DocumentIndex(cohere_api_key="test")
        assert idx.list_documents() == []

    def test_remove_nonexistent_document(self):
        idx = DocumentIndex(cohere_api_key="test")
        idx.remove_document("nonexistent")  # should not raise

    def test_remove_document_clears_chunks(self):
        idx = DocumentIndex(cohere_api_key="test")
        # Manually add chunks to bypass embedding
        chunk = Chunk(
            id="c1", doc_id="doc1", text="test", index=0,
            start_char=0, end_char=4
        )
        idx._chunks["c1"] = chunk
        idx._embeddings["c1"] = [1.0, 0.0, 0.0]
        idx._doc_chunks["doc1"] = ["c1"]

        assert idx.total_chunks == 1
        idx.remove_document("doc1")
        assert idx.total_chunks == 0
        assert idx.list_documents() == []

    def test_list_documents_after_manual_add(self):
        idx = DocumentIndex(cohere_api_key="test")
        idx._doc_chunks["doc_a"] = ["c1", "c2"]
        idx._doc_chunks["doc_b"] = ["c3"]

        docs = idx.list_documents()
        assert len(docs) == 2
        doc_ids = {d["doc_id"] for d in docs}
        assert doc_ids == {"doc_a", "doc_b"}


class TestSearchResult:
    def test_search_result_creation(self):
        chunk = Chunk(
            id="c1", doc_id="d1", text="hello", index=0,
            start_char=0, end_char=5
        )
        result = SearchResult(chunk=chunk, score=0.95)
        assert result.score == 0.95
        assert result.chunk.text == "hello"

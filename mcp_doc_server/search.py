"""
Semantic search over document chunks using Cohere embeddings.
"""

import os
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from mcp_doc_server.chunker import Chunk


@dataclass
class SearchResult:
    """A search result with relevance score."""
    chunk: Chunk
    score: float


class DocumentIndex:
    """
    In-memory vector index for document chunks.
    Uses Cohere Embed v3 for embeddings and cosine similarity for retrieval.
    """

    def __init__(self, cohere_api_key: Optional[str] = None):
        self._api_key = cohere_api_key or os.environ.get("COHERE_API_KEY")
        self._chunks: Dict[str, Chunk] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._doc_chunks: Dict[str, List[str]] = {}  # doc_id -> [chunk_ids]
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(self._api_key)
            except ImportError:
                raise ImportError("cohere is required: pip install cohere")
        return self._client

    async def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Add chunks to the index with embeddings.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = await self._embed(texts, input_type="search_document")

        for chunk, embedding in zip(chunks, embeddings):
            self._chunks[chunk.id] = chunk
            self._embeddings[chunk.id] = embedding

            if chunk.doc_id not in self._doc_chunks:
                self._doc_chunks[chunk.doc_id] = []
            self._doc_chunks[chunk.doc_id].append(chunk.id)

        return len(chunks)

    async def search(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Search chunks by semantic similarity.

        Args:
            query: Search query text.
            doc_id: Optional filter to search within a specific document.
            top_k: Number of results to return.

        Returns:
            List of SearchResult sorted by relevance.
        """
        query_embedding = (await self._embed([query], input_type="search_query"))[0]

        # Filter to specific doc if requested
        if doc_id and doc_id in self._doc_chunks:
            candidate_ids = self._doc_chunks[doc_id]
        else:
            candidate_ids = list(self._chunks.keys())

        if not candidate_ids:
            return []

        # Compute cosine similarities
        results = []
        for chunk_id in candidate_ids:
            if chunk_id not in self._embeddings:
                continue
            score = self._cosine_similarity(
                query_embedding, self._embeddings[chunk_id]
            )
            results.append(SearchResult(
                chunk=self._chunks[chunk_id],
                score=score,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def remove_document(self, doc_id: str):
        """Remove all chunks for a document from the index."""
        if doc_id not in self._doc_chunks:
            return
        for chunk_id in self._doc_chunks[doc_id]:
            self._chunks.pop(chunk_id, None)
            self._embeddings.pop(chunk_id, None)
        del self._doc_chunks[doc_id]

    def list_documents(self) -> List[Dict]:
        """List all indexed documents with chunk counts."""
        return [
            {"doc_id": doc_id, "chunk_count": len(chunk_ids)}
            for doc_id, chunk_ids in self._doc_chunks.items()
        ]

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    async def _embed(
        self, texts: List[str], input_type: str = "search_document"
    ) -> List[List[float]]:
        """Embed texts using Cohere Embed v3."""
        client = self._get_client()
        response = client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type=input_type,
        )
        return response.embeddings

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

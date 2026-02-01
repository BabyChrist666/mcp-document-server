"""
MCP Server implementation for document analysis.

Exposes document parsing, chunking, search, and summarization
as MCP tools and resources.
"""

import os
import json
import asyncio
import hashlib
from pathlib import Path
from typing import Optional

from mcp_doc_server.parsers import DocumentParserRegistry, ParsedDocument
from mcp_doc_server.chunker import chunk_text, chunk_by_pages
from mcp_doc_server.search import DocumentIndex


class DocumentAnalysisServer:
    """
    MCP-compliant document analysis server.

    Provides tools for extracting, chunking, searching, and
    summarizing documents via the Model Context Protocol.
    """

    def __init__(self):
        self.parser = DocumentParserRegistry()
        self.index = DocumentIndex()
        self._parsed_cache: dict[str, ParsedDocument] = {}

    def _doc_id(self, file_path: str) -> str:
        """Generate a stable document ID from file path."""
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]

    def _get_or_parse(self, file_path: str) -> ParsedDocument:
        """Parse a document, using cache if available."""
        doc_id = self._doc_id(file_path)
        if doc_id not in self._parsed_cache:
            self._parsed_cache[doc_id] = self.parser.parse(file_path)
        return self._parsed_cache[doc_id]

    # --- MCP Tool Implementations ---

    async def extract_text(self, file_path: str) -> dict:
        """
        Extract full text from a document.

        Args:
            file_path: Path to the document file.

        Returns:
            Dict with text content and metadata.
        """
        doc = self._get_or_parse(file_path)
        return {
            "text": doc.text,
            "word_count": doc.metadata.word_count,
            "pages": doc.metadata.pages,
            "file_type": doc.metadata.file_type,
        }

    async def chunk_document(
        self,
        file_path: str,
        chunk_size: int = 500,
        overlap: int = 50,
        index_chunks: bool = True,
    ) -> dict:
        """
        Split a document into overlapping chunks.

        Args:
            file_path: Path to the document file.
            chunk_size: Target chunk size in characters.
            overlap: Overlap between consecutive chunks.
            index_chunks: Whether to add chunks to the search index.

        Returns:
            Dict with chunks and document info.
        """
        doc = self._get_or_parse(file_path)
        doc_id = self._doc_id(file_path)

        if doc.pages:
            chunks = chunk_by_pages(
                doc.pages, doc_id, chunk_size=chunk_size, overlap=overlap
            )
        else:
            chunks = chunk_text(
                doc.text, doc_id, chunk_size=chunk_size, overlap=overlap
            )

        if index_chunks:
            await self.index.add_chunks(chunks)

        return {
            "doc_id": doc_id,
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "id": c.id,
                    "index": c.index,
                    "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                    "word_count": c.metadata.get("word_count", 0),
                    "page": c.metadata.get("page"),
                }
                for c in chunks
            ],
        }

    async def search_chunks(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> dict:
        """
        Semantic search across indexed document chunks.

        Args:
            query: Search query.
            doc_id: Optional document ID to scope search.
            top_k: Number of results.

        Returns:
            Dict with search results.
        """
        results = await self.index.search(query, doc_id=doc_id, top_k=top_k)
        return {
            "query": query,
            "results": [
                {
                    "chunk_id": r.chunk.id,
                    "doc_id": r.chunk.doc_id,
                    "score": round(r.score, 4),
                    "text": r.chunk.text,
                    "page": r.chunk.metadata.get("page"),
                }
                for r in results
            ],
            "total_indexed": self.index.total_chunks,
        }

    async def summarize_document(
        self,
        file_path: str,
        detail_level: str = "brief",
    ) -> dict:
        """
        Generate a document summary using Cohere.

        Args:
            file_path: Path to the document.
            detail_level: "brief" (1-2 sentences), "standard" (paragraph), "detailed" (multi-paragraph).

        Returns:
            Dict with summary and metadata.
        """
        doc = self._get_or_parse(file_path)

        length_map = {
            "brief": 100,
            "standard": 300,
            "detailed": 600,
        }
        max_tokens = length_map.get(detail_level, 300)

        # Truncate document if too long for context
        text = doc.text[:8000] if len(doc.text) > 8000 else doc.text

        try:
            import cohere
            client = cohere.Client(os.environ.get("COHERE_API_KEY"))
            response = client.chat(
                message=f"Summarize the following document in a {detail_level} manner:\n\n{text}",
                model="command-r-plus",
                max_tokens=max_tokens,
            )
            summary = response.text
        except Exception as e:
            # Fallback: extractive summary (first N sentences)
            sentences = text.split(". ")
            count = {"brief": 2, "standard": 5, "detailed": 10}.get(detail_level, 5)
            summary = ". ".join(sentences[:count]) + "."

        return {
            "summary": summary,
            "detail_level": detail_level,
            "source_word_count": doc.metadata.word_count,
            "file_type": doc.metadata.file_type,
        }

    async def get_metadata(self, file_path: str) -> dict:
        """
        Extract document metadata.

        Args:
            file_path: Path to the document.

        Returns:
            Dict with title, author, pages, word count, etc.
        """
        doc = self._get_or_parse(file_path)
        m = doc.metadata
        return {
            "title": m.title,
            "author": m.author,
            "pages": m.pages,
            "word_count": m.word_count,
            "file_type": m.file_type,
            "file_size_bytes": m.file_size_bytes,
            "file_path": m.file_path,
        }

    # --- MCP Protocol ---

    def get_tool_definitions(self) -> list:
        """Return MCP tool definitions."""
        return [
            {
                "name": "extract_text",
                "description": "Extract full text content from a document (PDF, DOCX, TXT, etc.)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the document file",
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "chunk_document",
                "description": "Split a document into overlapping chunks for RAG. Optionally indexes them for semantic search.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the document file",
                        },
                        "chunk_size": {
                            "type": "integer",
                            "description": "Target chunk size in characters (default: 500)",
                            "default": 500,
                        },
                        "overlap": {
                            "type": "integer",
                            "description": "Overlap between chunks in characters (default: 50)",
                            "default": 50,
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "search_chunks",
                "description": "Semantic search across indexed document chunks using Cohere embeddings",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "Optional document ID to scope search",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "summarize_document",
                "description": "Generate a summary of a document. Supports brief, standard, and detailed summaries.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the document file",
                        },
                        "detail_level": {
                            "type": "string",
                            "enum": ["brief", "standard", "detailed"],
                            "description": "Summary detail level (default: brief)",
                            "default": "brief",
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "get_metadata",
                "description": "Extract document metadata (title, author, page count, word count, file type)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the document file",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        ]

    async def handle_tool_call(self, name: str, arguments: dict) -> dict:
        """Route an MCP tool call to the appropriate handler."""
        handlers = {
            "extract_text": self.extract_text,
            "chunk_document": self.chunk_document,
            "search_chunks": self.search_chunks,
            "summarize_document": self.summarize_document,
            "get_metadata": self.get_metadata,
        }

        handler = handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown tool: {name}")

        return await handler(**arguments)


def main():
    """Run the MCP server over stdio."""
    import sys

    server = DocumentAnalysisServer()

    # Simple JSON-RPC over stdio implementation
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            method = request.get("method", "")

            if method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {"tools": server.get_tool_definitions()},
                }
            elif method == "tools/call":
                params = request.get("params", {})
                result = asyncio.run(
                    server.handle_tool_call(
                        params.get("name", ""),
                        params.get("arguments", {}),
                    )
                )
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {},
                }

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if "request" in dir() else None,
                "error": {"code": -32603, "message": str(e)},
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()

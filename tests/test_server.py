"""Tests for the MCP server tool definitions and routing."""

import os
import tempfile
import pytest
from mcp_doc_server.server import DocumentAnalysisServer


class TestToolDefinitions:
    def test_has_five_tools(self):
        server = DocumentAnalysisServer()
        tools = server.get_tool_definitions()
        assert len(tools) == 5

    def test_tool_names(self):
        server = DocumentAnalysisServer()
        tools = server.get_tool_definitions()
        names = {t["name"] for t in tools}
        assert names == {
            "extract_text",
            "chunk_document",
            "search_chunks",
            "summarize_document",
            "get_metadata",
        }

    def test_all_tools_have_schemas(self):
        server = DocumentAnalysisServer()
        for tool in server.get_tool_definitions():
            assert "inputSchema" in tool
            assert "properties" in tool["inputSchema"]
            assert "required" in tool["inputSchema"]

    def test_all_tools_have_descriptions(self):
        server = DocumentAnalysisServer()
        for tool in server.get_tool_definitions():
            assert tool.get("description"), f"{tool['name']} missing description"


class TestExtractText:
    @pytest.mark.asyncio
    async def test_extract_text_from_txt(self):
        server = DocumentAnalysisServer()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Sample document for testing extraction.")
            f.flush()

            result = await server.extract_text(f.name)
            assert "Sample document" in result["text"]
            assert result["word_count"] == 5
            assert result["file_type"] == "txt"

        os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_extract_text_file_not_found(self):
        server = DocumentAnalysisServer()
        with pytest.raises(FileNotFoundError):
            await server.extract_text("/nonexistent/file.txt")


class TestGetMetadata:
    @pytest.mark.asyncio
    async def test_metadata_from_txt(self):
        server = DocumentAnalysisServer()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Metadata test content here.")
            f.flush()

            meta = await server.get_metadata(f.name)
            assert meta["file_type"] == "txt"
            assert meta["word_count"] == 4
            assert meta["file_size_bytes"] > 0

        os.unlink(f.name)


class TestChunkDocument:
    @pytest.mark.asyncio
    async def test_chunk_txt_no_index(self):
        server = DocumentAnalysisServer()
        text = "Word " * 200
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            f.flush()

            result = await server.chunk_document(
                f.name, chunk_size=100, overlap=10, index_chunks=False
            )
            assert result["total_chunks"] > 1
            assert len(result["chunks"]) == result["total_chunks"]

        os.unlink(f.name)


class TestHandleToolCall:
    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self):
        server = DocumentAnalysisServer()
        with pytest.raises(ValueError, match="Unknown tool"):
            await server.handle_tool_call("nonexistent_tool", {})

    @pytest.mark.asyncio
    async def test_routes_to_extract_text(self):
        server = DocumentAnalysisServer()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Routing test.")
            f.flush()

            result = await server.handle_tool_call(
                "extract_text", {"file_path": f.name}
            )
            assert "Routing test" in result["text"]

        os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_routes_to_get_metadata(self):
        server = DocumentAnalysisServer()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Meta routing.")
            f.flush()

            result = await server.handle_tool_call(
                "get_metadata", {"file_path": f.name}
            )
            assert result["file_type"] == "txt"

        os.unlink(f.name)

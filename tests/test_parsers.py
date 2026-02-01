"""Tests for document parsers."""

import os
import tempfile
import pytest
from mcp_doc_server.parsers import (
    TextParser,
    PDFParser,
    DocxParser,
    DocumentParserRegistry,
)


class TestTextParser:
    def test_supports_txt(self):
        parser = TextParser()
        assert parser.supports("document.txt")
        assert parser.supports("README.md")
        assert parser.supports("data.json")
        assert parser.supports("config.yaml")

    def test_does_not_support_pdf(self):
        parser = TextParser()
        assert not parser.supports("document.pdf")
        assert not parser.supports("document.docx")

    def test_parse_txt(self):
        parser = TextParser()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello world. This is a test document.")
            f.flush()

            doc = parser.parse(f.name)
            assert "Hello world" in doc.text
            assert doc.metadata.word_count == 7
            assert doc.metadata.file_type == "txt"

        os.unlink(f.name)

    def test_parse_empty_file(self):
        parser = TextParser()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("")
            f.flush()

            doc = parser.parse(f.name)
            assert doc.text == ""
            assert doc.metadata.word_count == 0

        os.unlink(f.name)


class TestPDFParser:
    def test_supports_pdf(self):
        parser = PDFParser()
        assert parser.supports("document.pdf")
        assert parser.supports("REPORT.PDF")

    def test_does_not_support_txt(self):
        parser = PDFParser()
        assert not parser.supports("document.txt")


class TestDocxParser:
    def test_supports_docx(self):
        parser = DocxParser()
        assert parser.supports("document.docx")
        assert parser.supports("CONTRACT.DOCX")

    def test_does_not_support_pdf(self):
        parser = DocxParser()
        assert not parser.supports("document.pdf")


class TestDocumentParserRegistry:
    def test_supported_types(self):
        registry = DocumentParserRegistry()
        types = registry.supported_types()
        assert "pdf" in types
        assert "docx" in types
        assert "txt" in types

    def test_parse_txt_file(self):
        registry = DocumentParserRegistry()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Test content for parsing.")
            f.flush()

            doc = registry.parse(f.name)
            assert "Test content" in doc.text

        os.unlink(f.name)

    def test_parse_nonexistent_raises(self):
        registry = DocumentParserRegistry()
        with pytest.raises(FileNotFoundError):
            registry.parse("/nonexistent/file.txt")

    def test_unsupported_extension_raises(self):
        registry = DocumentParserRegistry()
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"data")
            f.flush()

            with pytest.raises(ValueError, match="Unsupported file type"):
                registry.parse(f.name)

        os.unlink(f.name)

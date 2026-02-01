# MCP Document Analysis Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that provides document analysis capabilities to LLM applications. Supports PDF, DOCX, and plaintext extraction with chunking, summarization, and semantic search.

## Features

- **Document Extraction** - Parse PDF, DOCX, and TXT files into structured text
- **Smart Chunking** - Split documents with configurable overlap for RAG pipelines
- **Semantic Search** - Embed and search document chunks using Cohere Embed v3
- **Summarization** - Generate document summaries with configurable detail level
- **Metadata Extraction** - Extract titles, authors, dates, page counts
- **MCP Protocol** - Full MCP compliance for integration with Claude, IDEs, and other MCP hosts

## Architecture

```
MCP Client (Claude, IDE, etc.)
        │
        │ MCP Protocol (JSON-RPC over stdio)
        │
        ▼
┌─────────────────────────────┐
│   MCP Document Server       │
├─────────────────────────────┤
│   Tools:                    │
│   ├── extract_text          │
│   ├── chunk_document        │
│   ├── search_chunks         │
│   ├── summarize_document    │
│   └── get_metadata          │
├─────────────────────────────┤
│   Resources:                │
│   ├── document://{path}     │
│   └── chunks://{doc_id}     │
├─────────────────────────────┤
│   Parsers:                  │
│   ├── PDFParser             │
│   ├── DocxParser            │
│   └── TextParser            │
└─────────────────────────────┘
```

## Quick Start

### Installation

```bash
git clone https://github.com/BabyChrist666/mcp-document-server.git
cd mcp-document-server

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Usage with Claude Desktop

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "document-analysis": {
      "command": "python",
      "args": ["-m", "mcp_doc_server"],
      "cwd": "/path/to/mcp-document-server"
    }
  }
}
```

### Usage with Claude Code CLI

```bash
claude --mcp-server "python -m mcp_doc_server"
```

## Tools

### `extract_text`

Extract full text from a document file.

```json
{
  "name": "extract_text",
  "arguments": {
    "file_path": "/path/to/document.pdf"
  }
}
```

### `chunk_document`

Split a document into overlapping chunks for RAG.

```json
{
  "name": "chunk_document",
  "arguments": {
    "file_path": "/path/to/document.pdf",
    "chunk_size": 500,
    "overlap": 50
  }
}
```

### `search_chunks`

Semantic search across document chunks.

```json
{
  "name": "search_chunks",
  "arguments": {
    "query": "What are the payment terms?",
    "doc_id": "contract_2024",
    "top_k": 5
  }
}
```

### `summarize_document`

Generate a summary of a document.

```json
{
  "name": "summarize_document",
  "arguments": {
    "file_path": "/path/to/report.pdf",
    "detail_level": "brief"
  }
}
```

### `get_metadata`

Extract document metadata (title, author, pages, etc.).

```json
{
  "name": "get_metadata",
  "arguments": {
    "file_path": "/path/to/document.pdf"
  }
}
```

## Configuration

| Variable | Description | Default |
|:---------|:------------|:--------|
| `COHERE_API_KEY` | Cohere API key for embeddings and generation | Required |
| `EMBEDDING_MODEL` | Cohere embedding model | `embed-english-v3.0` |
| `CHUNK_SIZE` | Default chunk size in characters | `500` |
| `CHUNK_OVERLAP` | Default overlap between chunks | `50` |

## Testing

```bash
pytest tests/ -v
```

## Tech Stack

- **Python 3.10+** - Runtime
- **MCP SDK** - Model Context Protocol implementation
- **Cohere** - Embeddings and generation
- **PyPDF2** - PDF parsing
- **python-docx** - DOCX parsing
- **Pydantic** - Data validation

## License

MIT

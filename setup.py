"""Setup for mcp-document-server."""

from setuptools import setup, find_packages

setup(
    name="mcp-document-server",
    version="0.1.0",
    description="MCP server for document analysis with Cohere embeddings",
    author="BabyChrist666",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "cohere>=5.0",
        "PyPDF2>=3.0",
        "python-docx>=1.0",
        "pydantic>=2.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-asyncio>=0.21"],
    },
    entry_points={
        "console_scripts": [
            "mcp-doc-server=mcp_doc_server.__main__:main",
        ],
    },
)

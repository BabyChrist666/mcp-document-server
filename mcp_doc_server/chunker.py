"""
Document chunking with configurable size and overlap.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import hashlib


@dataclass
class Chunk:
    """A single document chunk."""
    id: str
    doc_id: str
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separator: str = "\n",
) -> List[Chunk]:
    """
    Split text into overlapping chunks.

    Args:
        text: Full document text.
        doc_id: Identifier for the source document.
        chunk_size: Target chunk size in characters.
        overlap: Number of overlapping characters between chunks.
        separator: Preferred split point (tries to split at this boundary).

    Returns:
        List of Chunk objects.
    """
    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end, try to break at a separator
        if end < len(text):
            # Look for the last separator within the chunk
            last_sep = text.rfind(separator, start, end)
            if last_sep > start:
                end = last_sep + len(separator)

        chunk_text_content = text[start:end].strip()

        if chunk_text_content:
            chunk_id = hashlib.sha256(
                f"{doc_id}:{index}:{start}".encode()
            ).hexdigest()[:12]

            chunks.append(Chunk(
                id=chunk_id,
                doc_id=doc_id,
                text=chunk_text_content,
                index=index,
                start_char=start,
                end_char=end,
                metadata={
                    "chunk_size": len(chunk_text_content),
                    "word_count": len(chunk_text_content.split()),
                },
            ))
            index += 1

        # Move forward by chunk_size - overlap
        start = end - overlap
        if start >= len(text):
            break
        # Avoid infinite loops
        if start <= chunks[-1].start_char if chunks else start <= 0:
            start = end

    return chunks


def chunk_by_pages(
    pages: List[str],
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Chunk a document page by page, then sub-chunk if pages are large.

    Args:
        pages: List of page texts.
        doc_id: Document identifier.
        chunk_size: Max chunk size.
        overlap: Overlap between chunks.

    Returns:
        List of Chunk objects with page metadata.
    """
    all_chunks = []

    for page_num, page_text in enumerate(pages):
        if not page_text.strip():
            continue

        page_chunks = chunk_text(
            text=page_text,
            doc_id=f"{doc_id}_p{page_num}",
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in page_chunks:
            chunk.metadata["page"] = page_num + 1
            chunk.doc_id = doc_id

        all_chunks.extend(page_chunks)

    # Re-index
    for i, chunk in enumerate(all_chunks):
        chunk.index = i

    return all_chunks

"""Implement fixed-size and structure-aware chunking."""
from .models import Chunk, DocumentPage


def _make_chunk(page: DocumentPage, text: str, strategy: str, index: int) -> Chunk:
    metadata = dict(page.metadata)
    metadata.update({"chunk_index": index, "chunk_strategy": strategy})
    return Chunk(
        chunk_id=f"{page.document_id}:{page.page}:{strategy}:{index}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split page text into contiguous windows with a fixed overlap."""
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = page.text
    if not text or not text.strip():
        return []

    chunks = []
    step = size - overlap
    for index, start in enumerate(range(0, len(text), step)):
        window = text[start : start + size]
        if not window:
            break
        chunks.append(_make_chunk(page, window, "fixed_size", index))
        if start + size >= len(text):
            break
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraphs up to max_size, splitting longer paragraphs safely."""
    if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size <= 0:
        raise ValueError("max_size must be a positive integer")

    text = page.text
    if not text or not text.strip():
        return []

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    blocks = []
    for paragraph in paragraphs:
        remaining = paragraph
        while len(remaining) > max_size:
            # Prefer the last whitespace boundary in range; otherwise hard-split.
            split_at = remaining.rfind(" ", 0, max_size + 1)
            if split_at <= 0:
                split_at = max_size
            blocks.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
        if remaining:
            blocks.append(remaining)

    chunks = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = block
    if current:
        chunks.append(current)

    return [
        _make_chunk(page, chunk_text, "by_paragraph", index)
        for index, chunk_text in enumerate(chunks)
    ]

"""
chunker.py — Document chunking strategies.

WHY CHUNKING MATTERS: LLMs have context limits; Qdrant indexes fixed-size
vectors. We must split documents into chunks that are:
  1. Small enough to embed meaningfully (one idea per chunk)
  2. Large enough to contain useful context
  3. Overlapping so ideas spanning chunk boundaries aren't lost

RECURSIVE CHARACTER SPLITTING:
  Split by paragraph -> sentence -> word -> character, in that order.
  This respects document structure: only splits sentences when paragraphs
  are too long, only splits words when sentences are too long.

PARENT-CHILD CHUNKING:
  Large parent chunks (1024 tokens) for context.
  Small child chunks (256 tokens) for precise retrieval.
  Children store parent_id in metadata for lookup after retrieval.

INTERVIEW: "I use RecursiveCharacterTextSplitter because it's hierarchy-
aware — it tries to keep semantic units (paragraphs, sentences) intact.
Fixed-size splitting is simpler but cuts sentences mid-thought."

COMMON MISTAKE: chunk_overlap=0 causes context loss at boundaries.
Rule of thumb: overlap = 10-15% of chunk_size.

PERFORMANCE: For 1M token corpus, chunking takes ~2s. Not a bottleneck.
The bottleneck is embedding (50 chunks/sec on CPU bge-small).
"""

from __future__ import annotations
import re
from backend.config import get_settings


def recursive_split(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """
    Split text recursively by paragraph -> sentence -> character.
    Returns list of text chunks within chunk_size limit.
    """
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    # Separators tried in order: paragraph, sentence, word, char
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps or len(text) <= size:
            return [text] if text.strip() else []

        sep = seps[0]
        splits = text.split(sep) if sep else list(text)
        chunks: list[str] = []
        current = ""

        for piece in splits:
            piece = piece.strip()
            if not piece:
                continue
            candidate = (current + sep + piece).strip() if current else piece
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(piece) > size:
                    # Recursively split the oversized piece
                    chunks.extend(_split(piece, seps[1:]))
                    current = ""
                else:
                    current = piece

        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, separators)

    # Apply overlap by prepending tail of previous chunk
    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped: list[str] = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i - 1][-overlap:]
        overlapped.append((prev_tail + " " + raw_chunks[i]).strip())

    return overlapped


def create_parent_child_chunks(
    text: str,
    parent_size: int = 1024,
    child_size: int = 256,
    overlap: int = 32,
) -> tuple[list[str], list[list[str]]]:
    """
    Create parent chunks and their child sub-chunks.

    Returns:
        (parent_texts, child_texts_per_parent)
        parent_texts[i] corresponds to child_texts_per_parent[i]
    """
    parents = recursive_split(text, chunk_size=parent_size, chunk_overlap=overlap)
    children_per_parent = [
        recursive_split(parent, chunk_size=child_size, chunk_overlap=overlap)
        for parent in parents
    ]
    return parents, children_per_parent

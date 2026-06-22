"""
document_loader.py — Load documents from files or raw text.

Supports: .txt, .md, .pdf (via pdfminer), .html (via html.parser)
Falls back gracefully if optional deps not installed.

INTERVIEW: "I keep the loader generic — a DocumentLoader protocol
with a load() method. Adding a new source (S3, Confluence, Notion)
is a new class, not a modification to the pipeline."
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_text_file(path: str | Path) -> str:
    """Load a plain text or markdown file."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def load_html(html: str) -> str:
    """Strip HTML tags and return plain text."""
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts: list[str] = []
        def handle_data(self, data):
            self.parts.append(data)

    s = _Stripper()
    s.feed(html)
    return " ".join(s.parts)


def load_pdf(path: str | Path) -> str:
    """Extract text from PDF using pdfminer.six if available."""
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    except ImportError:
        logger.warning("pdfminer.six not installed. Install it to read PDFs.")
        return ""


def load_document(path: str | Path) -> tuple[str, dict[str, Any]]:
    """
    Load a document from disk. Returns (text, metadata).
    Metadata includes source path, filename, extension.
    """
    p = Path(path)
    meta: dict[str, Any] = {
        "source": str(p),
        "title": p.stem,
        "extension": p.suffix.lower(),
    }

    ext = p.suffix.lower()
    if ext in (".txt", ".md", ".rst"):
        text = load_text_file(p)
    elif ext == ".pdf":
        text = load_pdf(p)
    elif ext in (".html", ".htm"):
        text = load_html(p.read_text(encoding="utf-8", errors="replace"))
    else:
        logger.warning("Unsupported extension %s, loading as text", ext)
        text = load_text_file(p)

    return text, meta


def load_documents_from_dir(directory: str | Path) -> list[tuple[str, dict[str, Any]]]:
    """Load all supported documents from a directory."""
    SUPPORTED = {".txt", ".md", ".pdf", ".html", ".htm", ".rst"}
    docs = []
    for p in Path(directory).rglob("*"):
        if p.suffix.lower() in SUPPORTED and p.is_file():
            try:
                text, meta = load_document(p)
                if text.strip():
                    docs.append((text, meta))
            except Exception as e:
                logger.error("Failed to load %s: %s", p, e)
    logger.info("Loaded %d documents from %s", len(docs), directory)
    return docs

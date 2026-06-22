"""
indexer.py — Full ingestion pipeline: load -> chunk -> embed -> index.

DATA FLOW:
  raw text
    -> recursive_split() -> chunks
    -> embed_documents() -> vectors
    -> qdrant upsert    -> searchable
    -> bm25 add_docs    -> BM25 index updated
    -> parent-child     -> parent collection updated

INTERVIEW: "The indexer is the write path of RAG. I keep it separate
from the retrieval (read path) so they can scale independently. In
production the indexer would run as a background worker (Celery/RQ)
triggered by document upload events."

COMMON MISTAKE: Indexing into Qdrant but forgetting to update the BM25
index — hybrid search then misses all new documents.

PERFORMANCE: The bottleneck is embedding (CPU). For large corpora,
use larger batch sizes or switch to GPU. Qdrant upsert is fast (<50ms
for 1000 points).
"""

import logging
import uuid
from typing import Any
from backend.ingestion.document_loader import load_document, load_documents_from_dir
from backend.ingestion.chunker import recursive_split, create_parent_child_chunks
from backend.retrieval.dense_retriever import upsert_documents, ensure_collection
from backend.retrieval.bm25_retriever import add_documents as bm25_add
from backend.retrieval.parent_child import index_parent_child
from backend.config import get_settings

logger = logging.getLogger(__name__)


def ingest_texts(
    texts: list[str],
    metadatas: list[dict[str, Any]] | None = None,
    use_parent_child: bool = False,
) -> int:
    """
    Full ingestion pipeline for raw texts.
    Chunks, embeds, and indexes into Qdrant + BM25.
    Returns total chunks indexed.
    """
    settings = get_settings()
    metas = metadatas or [{} for _ in texts]
    ensure_collection()
    ensure_collection(settings.qdrant_parent_collection)

    all_chunks: list[str] = []
    all_chunk_metas: list[dict[str, Any]] = []

    for text, meta in zip(texts, metas):
        if use_parent_child:
            parents, children_per_parent = create_parent_child_chunks(text)
            index_parent_child(
                parent_texts=parents,
                parent_metadatas=[meta] * len(parents),
                child_texts_per_parent=children_per_parent,
            )
            for children in children_per_parent:
                all_chunks.extend(children)
                all_chunk_metas.extend([meta] * len(children))
        else:
            chunks = recursive_split(text)
            all_chunks.extend(chunks)
            all_chunk_metas.extend([meta] * len(chunks))

    if not all_chunks:
        logger.warning("No chunks produced from %d texts", len(texts))
        return 0

    # Index into Qdrant (dense)
    n = upsert_documents(all_chunks, all_chunk_metas)

    # Update BM25 index
    bm25_add(all_chunks, all_chunk_metas)

    logger.info("Ingestion complete: %d source texts -> %d chunks indexed", len(texts), n)
    return n


def ingest_file(path: str, use_parent_child: bool = False) -> int:
    """Load a single file and ingest it."""
    text, meta = load_document(path)
    return ingest_texts([text], [meta], use_parent_child=use_parent_child)


def ingest_directory(directory: str, use_parent_child: bool = False) -> int:
    """Load all documents from a directory and ingest them."""
    docs = load_documents_from_dir(directory)
    if not docs:
        return 0
    texts, metas = zip(*docs)
    return ingest_texts(list(texts), list(metas), use_parent_child=use_parent_child)

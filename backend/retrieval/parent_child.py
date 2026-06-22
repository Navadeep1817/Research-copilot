"""
parent_child.py — Parent-Child (Small-to-Big) Retrieval.

THE PROBLEM WITH FIXED CHUNKING:
  Small chunks (128 tokens): good for precise retrieval, but lack context
  Large chunks (1024 tokens): good for context, but embed poorly (too diffuse)

PARENT-CHILD SOLUTION:
  Index small "child" chunks for retrieval precision.
  Store large "parent" chunks for context richness.
  Retrieve by child similarity -> return parent content.

EXAMPLE:
  Parent chunk: entire section about "Transformer Architecture" (1000 tokens)
  Child chunks: each paragraph (~128 tokens)
  
  Query "what is multi-head attention?" retrieves the paragraph
  about multi-head attention (child), but returns the full section
  (parent) to the LLM for answering — giving it surrounding context.

INTERVIEW: "Parent-child retrieval solves the precision-context tradeoff.
Small chunks are precise retrieval units; large chunks give the LLM
the context it needs to answer well.  LlamaIndex calls this
'sentence window retrieval' when the parent is a sliding window."

COMMON MISTAKE: Not storing parent IDs in child chunk metadata. Without
the parent_id link you cannot look up the parent document after retrieval.

ALTERNATIVES: Sentence window retrieval (fixed-window parents),
proposition indexing (each chunk = one atomic fact).
"""

import logging
from typing import Any
from backend.config import get_settings
from backend.models.schemas import SourceDocument
from backend.retrieval.dense_retriever import dense_search, upsert_documents, get_qdrant_client

logger = logging.getLogger(__name__)


def index_parent_child(
    parent_texts: list[str],
    parent_metadatas: list[dict[str, Any]],
    child_texts_per_parent: list[list[str]],
) -> int:
    """
    Index parent and child chunks into separate Qdrant collections.
    
    Parent chunks go into QDRANT_PARENT_COLLECTION (with full text).
    Child chunks go into QDRANT_COLLECTION_NAME (with parent_id reference).
    
    Returns total number of child chunks indexed.
    """
    import uuid
    settings = get_settings()

    total = 0
    for i, (parent_text, parent_meta, children) in enumerate(
        zip(parent_texts, parent_metadatas, child_texts_per_parent)
    ):
        parent_id = str(uuid.uuid4())

        # Index parent chunk (we search by parent_id, not by vector)
        upsert_documents(
            texts=[parent_text],
            metadatas=[{**parent_meta, "is_parent": True}],
            ids=[parent_id],
            collection_name=settings.qdrant_parent_collection,
        )

        # Index child chunks with reference to parent
        child_metas = [
            {**parent_meta, "parent_id": parent_id, "is_child": True}
            for _ in children
        ]
        upsert_documents(
            texts=children,
            metadatas=child_metas,
            collection_name=settings.qdrant_collection_name,
        )
        total += len(children)

    logger.info("Parent-child indexing: %d parents, %d total children", len(parent_texts), total)
    return total


def parent_child_search(query: str, top_k: int | None = None) -> list[SourceDocument]:
    """
    1. Search child collection with dense retrieval.
    2. Look up parent documents via parent_id metadata.
    3. Return parent document content (richer context).
    """
    settings = get_settings()
    k = top_k or settings.top_k_retrieval
    client = get_qdrant_client()

    # Step 1: Retrieve child chunks
    child_results = dense_search(query, top_k=k * 2)

    # Step 2: Collect unique parent IDs
    seen_parents: set[str] = set()
    parent_docs: list[SourceDocument] = []

    for child in child_results:
        parent_id = child.metadata.get("parent_id")
        if not parent_id or parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        # Step 3: Fetch parent from parent collection
        try:
            results = client.retrieve(
                collection_name=settings.qdrant_parent_collection,
                ids=[parent_id],
                with_payload=True,
            )
            if results:
                payload = results[0].payload or {}
                parent_docs.append(SourceDocument(
                    doc_id=parent_id,
                    content=payload.get("content", child.content),
                    source=payload.get("source", child.source),
                    title=payload.get("title", child.title),
                    score=child.score,
                    strategy="parent_child",
                    metadata=payload,
                ))
        except Exception as e:
            logger.warning("Failed to fetch parent %s: %s", parent_id, e)
            # Fall back to child content
            child.strategy = "parent_child"
            parent_docs.append(child)

        if len(parent_docs) >= k:
            break

    logger.debug("Parent-child search: %d child hits -> %d parent docs", len(child_results), len(parent_docs))
    return parent_docs

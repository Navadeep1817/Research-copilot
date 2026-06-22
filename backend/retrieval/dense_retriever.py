"""
dense_retriever.py — Qdrant ANN vector search.

Fixed for qdrant-client >= 1.7 where client.search() was replaced by
client.query_points(). The new API uses QueryResponse instead of ScoredPoint.
"""

import logging
import uuid
from typing import Any
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    NamedVector, Query
)
from backend.config import get_settings
from backend.models.schemas import SourceDocument

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    path = settings.qdrant_storage_path
    path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(path))
    logger.info("Qdrant client at %s", path)
    return client


def ensure_collection(collection_name: str | None = None) -> None:
    settings = get_settings()
    name = collection_name or settings.qdrant_collection_name
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection: %s", name)


def upsert_documents(
    texts: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str] | None = None,
    collection_name: str | None = None,
) -> int:
    from backend.retrieval.embeddings import embed_documents
    settings = get_settings()
    name = collection_name or settings.qdrant_collection_name
    ensure_collection(name)
    client = get_qdrant_client()

    vectors = embed_documents(texts)
    points = [
        PointStruct(
            id=ids[i] if ids else str(uuid.uuid4()),
            vector=vectors[i],
            payload={"content": texts[i], **(metadatas[i] if i < len(metadatas) else {})},
        )
        for i in range(len(texts))
    ]
    client.upsert(collection_name=name, points=points)
    logger.info("Upserted %d points into %s", len(points), name)
    return len(points)


def dense_search(
    query: str,
    top_k: int | None = None,
    collection_name: str | None = None,
    filter_conditions: dict[str, Any] | None = None,
) -> list[SourceDocument]:
    """
    Vector similarity search using the new query_points() API
    (qdrant-client >= 1.7).
    Falls back to search() for older versions.
    """
    from backend.retrieval.embeddings import embed_query
    settings = get_settings()
    k = top_k or settings.top_k_retrieval
    name = collection_name or settings.qdrant_collection_name
    client = get_qdrant_client()

    # Check collection exists
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        logger.warning("Collection %s does not exist — returning empty", name)
        return []

    query_vector = embed_query(query)

    qdrant_filter = None
    if filter_conditions:
        qdrant_filter = Filter(
            must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_conditions.items()
            ]
        )

    # ── Try new API first (qdrant-client >= 1.7) ──────────────────────────
    try:
        response = client.query_points(
            collection_name=name,
            query=query_vector,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        # response.points is a list of ScoredPoint
        hits = response.points
    except AttributeError:
        # ── Fall back to legacy search() API ──────────────────────────────
        logger.debug("query_points not available, falling back to search()")
        hits = client.search(
            collection_name=name,
            query_vector=query_vector,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

    docs = []
    for hit in hits:
        payload = hit.payload or {}
        docs.append(SourceDocument(
            doc_id=str(hit.id),
            content=payload.get("content", ""),
            source=payload.get("source", ""),
            title=payload.get("title", ""),
            score=hit.score,
            strategy="dense",
            metadata={
                k: v for k, v in payload.items()
                if k not in ("content", "source", "title")
            },
        ))

    logger.debug("Dense search: %d results for query: %.60s", len(docs), query)
    return docs


def retrieve_by_id(point_id: str, collection_name: str | None = None) -> SourceDocument | None:
    """Fetch a single point by ID (used by parent-child retrieval)."""
    settings = get_settings()
    name = collection_name or settings.qdrant_collection_name
    client = get_qdrant_client()

    try:
        results = client.retrieve(
            collection_name=name,
            ids=[point_id],
            with_payload=True,
        )
        if not results:
            return None
        payload = results[0].payload or {}
        return SourceDocument(
            doc_id=point_id,
            content=payload.get("content", ""),
            source=payload.get("source", ""),
            title=payload.get("title", ""),
            score=1.0,
            strategy="parent_child",
            metadata=payload,
        )
    except Exception as e:
        logger.error("retrieve_by_id failed for %s: %s", point_id, e)
        return None 
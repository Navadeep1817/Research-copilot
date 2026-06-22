"""
metadata_filter.py — Structured metadata filtering on Qdrant payloads.

WHY METADATA FILTERING: Vector similarity alone cannot answer
structured questions like "papers published after 2023" or
"documents from source=arxiv". Metadata filters are applied BEFORE
or DURING vector search, drastically reducing the candidate set.

QDRANT FILTER TYPES:
  - MatchValue: exact match  {field: "arxiv"}
  - MatchAny: enum match     {field: ["arxiv", "pubmed"]}
  - Range: numeric range     {year: {gte: 2022, lte: 2024}}
  - IsEmpty / IsNull: null checks

PRE-FILTERING vs POST-FILTERING:
  Qdrant applies filters during HNSW traversal (pre-filtering by default
  when filter selectivity is high) — much faster than fetching all and
  filtering in Python.

INTERVIEW: "Metadata filtering is critical for production RAG because
users often have implicit structural constraints — they want recent docs,
docs from a specific source, or docs in a certain category. Encoding
these as Qdrant payload filters means the vector index only scores
relevant candidates."

COMMON MISTAKE: Storing all metadata as strings and doing regex
matching in Python after retrieval — this is O(n) and misses the
Qdrant index optimisation.

SCALABILITY: Qdrant supports payload indexes (create_payload_index)
which turn O(n) filter scans into O(log n) lookups for high-cardinality
fields like date or author.
"""

import logging
from typing import Any
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range
from backend.config import get_settings
from backend.models.schemas import SourceDocument
from backend.retrieval.dense_retriever import dense_search, get_qdrant_client
from backend.retrieval.embeddings import embed_query

logger = logging.getLogger(__name__)


def build_qdrant_filter(filter_dict: dict[str, Any]) -> Filter | None:
    """
    Convert a simple filter dict into a Qdrant Filter object.

    Supported formats:
        {"source": "arxiv"}                    -> exact match
        {"source": ["arxiv", "pubmed"]}        -> match any
        {"year": {"gte": 2022, "lte": 2024}}   -> range filter
    """
    if not filter_dict:
        return None

    conditions = []
    for field, value in filter_dict.items():
        if isinstance(value, list):
            conditions.append(FieldCondition(key=field, match=MatchAny(any=value)))
        elif isinstance(value, dict):
            # Range filter
            range_params = {}
            if "gte" in value:
                range_params["gte"] = value["gte"]
            if "lte" in value:
                range_params["lte"] = value["lte"]
            if "gt" in value:
                range_params["gt"] = value["gt"]
            if "lt" in value:
                range_params["lt"] = value["lt"]
            conditions.append(FieldCondition(key=field, range=Range(**range_params)))
        else:
            conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))

    return Filter(must=conditions) if conditions else None


def metadata_filter_search(
    query: str,
    filter_dict: dict[str, Any],
    top_k: int | None = None,
) -> list[SourceDocument]:
    """
    Dense search with Qdrant payload filters.

    Args:
        query: Search query
        filter_dict: Dict of field -> value constraints
        top_k: Number of results

    Example:
        metadata_filter_search(
            "transformer architecture",
            {"source": "arxiv", "year": {"gte": 2022}},
            top_k=5,
        )
    """
    settings = get_settings()
    k = top_k or settings.top_k_retrieval

    results = dense_search(
        query=query,
        top_k=k,
        filter_conditions=filter_dict,
    )
    for doc in results:
        doc.strategy = "metadata_filter"

    logger.debug(
        "Metadata filter search: filter=%s returned %d docs",
        filter_dict, len(results),
    )
    return results

"""
hybrid_retriever.py — Reciprocal Rank Fusion (RRF) of dense + BM25.

WHY HYBRID: BEIR benchmark shows hybrid retrieval outperforms either
dense or sparse alone on 12/18 datasets.  The intuition: dense captures
semantic similarity, BM25 captures exact keyword matches.

WHY RRF (not score interpolation):
  Dense scores are cosine similarities in [-1,1].
  BM25 scores are unnormalised TF-IDF-style floats with no upper bound.
  You CANNOT add them directly.  RRF uses rank positions, which are
  comparable across systems.

  RRF(d) = Σ_retriever  1 / (k + rank_i(d))
  where k=60 is an empirical smoothing constant (from the original paper).

  A document ranked #1 by both retrievers gets ~2*(1/61) = 0.033
  A document ranked #50 by both gets ~2*(1/110) = 0.018
  k=60 prevents top-ranked documents from dominating excessively.

INTERVIEW: "I chose RRF over weighted score interpolation because it's
parameter-free — no need to tune alpha weights for each dataset.
Empirically, RRF matches tuned interpolation within 1-2% NDCG."

COMMON MISTAKE: Including documents with zero scores from one retriever
before computing RRF.  Filter out zero-score docs first.

ALTERNATIVES: Convex combination (alpha * dense + (1-alpha) * bm25),
CombMNZ (sum of scores * number of non-zero retrievers contributing).
"""

import logging
from backend.config import get_settings
from backend.models.schemas import SourceDocument
from backend.retrieval.dense_retriever import dense_search
from backend.retrieval.bm25_retriever import bm25_search

logger = logging.getLogger(__name__)

RRF_K = 60  # smoothing constant from original RRF paper (Cormack et al. 2009)


def reciprocal_rank_fusion(
    result_lists: list[list[SourceDocument]],
    k: int = RRF_K,
) -> list[SourceDocument]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        result_lists: Each inner list is a ranked list from one retriever.
        k: RRF smoothing constant.

    Returns:
        Single merged list sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, SourceDocument] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            key = doc.content[:200]  # use content prefix as dedup key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc

    merged = sorted(doc_map.keys(), key=lambda k: scores[k], reverse=True)
    result = []
    for key in merged:
        doc = doc_map[key]
        doc.score = scores[key]
        doc.strategy = "hybrid_rrf"
        result.append(doc)

    return result


def hybrid_search(
    query: str,
    top_k: int | None = None,
    filter_conditions: dict | None = None,
) -> list[SourceDocument]:
    """
    Run dense + BM25 retrieval and fuse with RRF.

    We over-fetch (2x top_k) from each retriever before fusion so the
    fusion has enough candidates to work with.
    """
    settings = get_settings()
    k = top_k or settings.top_k_retrieval
    fetch_k = k * 2  # over-fetch before fusion

    dense_results = dense_search(query, top_k=fetch_k, filter_conditions=filter_conditions)
    bm25_results  = bm25_search(query, top_k=fetch_k)

    fused = reciprocal_rank_fusion([dense_results, bm25_results])
    logger.debug(
        "Hybrid RRF: dense=%d bm25=%d -> fused=%d (returning top %d)",
        len(dense_results), len(bm25_results), len(fused), k,
    )
    return fused[:k]

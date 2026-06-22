"""
multi_query.py — Generate N query variants, retrieve in parallel, fuse.

WHY MULTI-QUERY: A single query may miss relevant documents due to
vocabulary mismatch.  Generating 3-5 alternative phrasings and fusing
results dramatically improves recall.

EXAMPLE:
  Original: "LLM hallucination prevention techniques"
  Variants:
    "methods to reduce factual errors in large language models"
    "how to make AI systems more reliable and accurate"
    "retrieval augmented generation to ground LLM responses"

Each variant retrieves different but relevant documents.  RRF fusion
ensures we return the most consistently-retrieved ones.

INTERVIEW: "Multi-query is a query-side ensemble.  It trades LLM
inference cost (N extra calls) for higher recall.  In production I'd
cache variants by query hash and run the retrievals in parallel with
asyncio.gather."

COMMON MISTAKE: Generating too many variants (>5) — latency grows
linearly and diminishing returns set in after 3-4 variants.
"""

import asyncio
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.config import get_settings
from backend.models.schemas import SourceDocument
from backend.retrieval.dense_retriever import dense_search
from backend.retrieval.hybrid_retriever import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Generate {n} different search query variants for the given question. "
        "Each variant should use different vocabulary but seek the same information. "
        "Return ONLY the queries, one per line, no numbering or bullets."
    )),
    ("human", "Question: {query}"),
])


def generate_query_variants(query: str, n: int = 3) -> list[str]:
    """Use LLM to generate N alternative query phrasings."""
    settings = get_settings()
    try:
        llm = ChatGroq(model=settings.groq_model_name, temperature=0.7, api_key=settings.groq_api_key)
        chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
        output = chain.invoke({"query": query, "n": n})
        variants = [line.strip() for line in output.strip().split("\n") if line.strip()]
        logger.debug("Generated %d query variants", len(variants))
        return ([query] + variants)[:n + 1]  # always include original
    except Exception as e:
        logger.warning("Multi-query generation failed: %s", e)
        return [query]


def multi_query_search(query: str, top_k: int | None = None, n_variants: int = 3) -> list[SourceDocument]:
    """
    Generate query variants, retrieve for each, fuse with RRF.
    All retrievals run sequentially (async version in agents uses asyncio.gather).
    """
    settings = get_settings()
    k = top_k or settings.top_k_retrieval
    variants = generate_query_variants(query, n=n_variants)

    all_results = []
    for variant in variants:
        results = dense_search(variant, top_k=k)
        if results:
            all_results.append(results)

    if not all_results:
        return []

    fused = reciprocal_rank_fusion(all_results)
    for doc in fused:
        doc.strategy = "multi_query"
    return fused[:k]

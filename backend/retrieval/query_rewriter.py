"""
query_rewriter.py — LLM-based query rewriting for better retrieval.

WHY QUERY REWRITING: Users write conversational, ambiguous queries.
"What did they say about it?" is useless for retrieval.  The LLM
rewrites it into a self-contained, retrieval-optimised query.

THREE REWRITING STRATEGIES:
  1. Clarify: expand abbreviations, resolve pronouns, add context
  2. Decompose: break complex question into simpler retrieval queries
  3. Rephrase: generate alternative phrasings (see multi_query.py)

INTERVIEW: "Query rewriting is a pre-retrieval technique.  The user's
raw query is often a poor search query — it may contain pronouns,
assumed context, or be too broad.  The LLM acts as a query
reformulation layer before hitting the retrieval index."

COMMON MISTAKE: Using the rewritten query for the final answer but
returning sources found with the original query — the mismatch hurts
faithfulness scores.  Always use the rewritten query end-to-end.

PERFORMANCE: One extra LLM call per query adds ~300ms latency on Groq.
Cache rewrites by query hash for repeated questions.
"""

import logging
from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.config import get_settings

logger = logging.getLogger(__name__)

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert at reformulating research queries for optimal "
        "retrieval from a knowledge base.  Given a user query, rewrite it "
        "to be more specific, self-contained, and retrieval-friendly.\n"
        "Rules:\n"
        "- Expand abbreviations\n"
        "- Remove ambiguous pronouns\n"
        "- Add domain-specific terminology\n"
        "- Keep it under 50 words\n"
        "Return ONLY the rewritten query, nothing else."
    )),
    ("human", "Original query: {query}"),
])


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model_name,
        temperature=0.0,    # deterministic for query rewriting
        api_key=settings.groq_api_key,
    )


def rewrite_query(query: str) -> str:
    """
    Rewrite a query for improved retrieval.
    Falls back to original query if LLM call fails.
    """
    try:
        chain = REWRITE_PROMPT | _get_llm() | StrOutputParser()
        rewritten = chain.invoke({"query": query}).strip()
        logger.debug("Query rewrite: [%s] -> [%s]", query[:60], rewritten[:60])
        return rewritten or query
    except Exception as e:
        logger.warning("Query rewrite failed (%s), using original", e)
        return query

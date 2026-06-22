"""
hyde.py — Hypothetical Document Embeddings (Gao et al. 2022).

WHY HYDE: Short, abstract queries embed poorly because they are
structurally different from the dense, detailed documents in the index.
HyDE asks the LLM to write a *hypothetical* answer document, then
embeds THAT — which is stylistically similar to real index documents.

ALGORITHM:
  1. LLM generates a hypothetical answer paragraph (~100 words)
  2. Embed the hypothetical answer (NOT the query)
  3. Use that embedding to search Qdrant
  4. Return real documents similar to the hypothetical answer

EXAMPLE:
  Query: "How does attention mechanism work?"
  HyDE generates: "The attention mechanism in transformers works by
  computing query, key, and value projections..."
  This hypothetical text is more similar to real academic documents
  than the short query "How does attention mechanism work?" is.

INTERVIEW: "HyDE improves recall for short queries by bridging the
distribution gap between user queries and index documents.  The key
insight is that the hypothetical document is a better query vector
than the actual query string."

COMMON MISTAKE: Using HyDE for factual queries where you need exact
answers — the LLM might hallucinate, and you end up retrieving
documents about the hallucinated answer instead of the real one.
Best for conceptual/explanatory questions.

PAPER: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
Gao et al., ACL 2023.
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.config import get_settings
from backend.models.schemas import SourceDocument
from backend.retrieval.embeddings import embed_query
from backend.retrieval.dense_retriever import dense_search

logger = logging.getLogger(__name__)

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Write a short, factual paragraph (~100 words) that directly answers "
        "the given question. Write as if you are a knowledgeable expert writing "
        "a document that would be stored in a research database. "
        "Focus on being informative and using domain-appropriate vocabulary. "
        "Do NOT add caveats like 'I don't know' — write a confident hypothetical answer."
    )),
    ("human", "Question: {query}"),
])


def generate_hypothetical_document(query: str) -> str:
    """Ask the LLM to write a hypothetical answer document."""
    settings = get_settings()
    try:
        llm = ChatGroq(model=settings.groq_model_name, temperature=0.5, api_key=settings.groq_api_key)
        chain = HYDE_PROMPT | llm | StrOutputParser()
        hyp_doc = chain.invoke({"query": query}).strip()
        logger.debug("HyDE generated %d chars for query: %.60s", len(hyp_doc), query)
        return hyp_doc
    except Exception as e:
        logger.warning("HyDE generation failed: %s", e)
        return query  # fall back to original query


def hyde_search(query: str, top_k: int | None = None) -> list[SourceDocument]:
    """
    HyDE retrieval: embed hypothetical doc, search, return real docs.
    """
    settings = get_settings()
    k = top_k or settings.top_k_retrieval

    # Generate and embed the hypothetical document
    hyp_doc = generate_hypothetical_document(query)

    # Use dense search but with the hypothetical document as the query
    # We abuse dense_search by passing hyp_doc as the query string
    # (it will be embedded — which is exactly what we want)
    results = dense_search(hyp_doc, top_k=k)
    for doc in results:
        doc.strategy = "hyde"

    logger.debug("HyDE search returned %d results", len(results))
    return results

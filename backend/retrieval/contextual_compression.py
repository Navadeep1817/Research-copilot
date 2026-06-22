"""
contextual_compression.py — LLM-based contextual compression.

THE PROBLEM: Retrieved chunks often contain irrelevant sentences
surrounding the actually-relevant passage. Sending 512-token chunks
to the LLM wastes context window and dilutes the relevant signal.

SOLUTION: After retrieval, pass each chunk through an LLM extractor
that pulls out ONLY the sentences relevant to the query.

PIPELINE:
  raw chunk (512 tokens) -> LLM extractor -> compressed passage (50-150 tokens)

INTERVIEW: "Contextual compression is a post-retrieval technique that
reduces noise in the context window. Instead of giving the LLM 10 full
chunks, I give it 10 surgically extracted passages. This improves
faithfulness scores because the LLM is less likely to get distracted
by irrelevant content in the same chunk."

COMMON MISTAKE: Running compression on every chunk regardless of
relevance score — wastes LLM calls. Only compress top-K after reranking.

PERFORMANCE: Each chunk = 1 LLM call. With top_k=5, that is 5 extra
calls. Cache by (query_hash, chunk_hash) to avoid repeat work.

ALTERNATIVES: LLMLingua (token-level compression, faster), extractive
summarization with transformers (no LLM needed but less precise).
"""

import logging
import hashlib
from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.config import get_settings
from backend.models.schemas import SourceDocument

logger = logging.getLogger(__name__)

COMPRESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert at extracting relevant information from documents. "
        "Given a query and a document chunk, extract ONLY the sentences or "
        "phrases that are directly relevant to answering the query. "
        "If nothing is relevant, respond with exactly: NOT_RELEVANT. "
        "Do not add explanations — return only the extracted text."
    )),
    ("human", "Query: {query}\n\nDocument chunk:\n{chunk}"),
])

# Simple in-memory cache: (query_hash + chunk_hash) -> compressed text
_compression_cache: dict[str, str] = {}


def _cache_key(query: str, chunk: str) -> str:
    return hashlib.md5((query + chunk).encode()).hexdigest()


def compress_document(query: str, doc: SourceDocument) -> SourceDocument | None:
    """
    Extract query-relevant sentences from a single document.
    Returns None if no relevant content is found.
    """
    settings = get_settings()
    key = _cache_key(query, doc.content)

    if key in _compression_cache:
        compressed = _compression_cache[key]
    else:
        try:
            llm = ChatGroq(model=settings.groq_model_name, temperature=0.0, api_key=settings.groq_api_key)
            chain = COMPRESS_PROMPT | llm | StrOutputParser()
            compressed = chain.invoke({"query": query, "chunk": doc.content}).strip()
            _compression_cache[key] = compressed
        except Exception as e:
            logger.warning("Compression failed for doc %s: %s", doc.doc_id, e)
            return doc  # return uncompressed on failure

    if compressed == "NOT_RELEVANT" or not compressed:
        return None

    compressed_doc = doc.model_copy()
    compressed_doc.content = compressed
    compressed_doc.strategy = "contextual_compress"
    return compressed_doc


def contextual_compression_search(
    query: str,
    base_docs: list[SourceDocument],
) -> list[SourceDocument]:
    """
    Apply contextual compression to a list of already-retrieved documents.
    Filters out irrelevant chunks. Always call AFTER retrieval + reranking.

    Args:
        query: The research question
        base_docs: Pre-retrieved documents to compress

    Returns:
        Compressed documents (fewer, shorter, more relevant)
    """
    compressed = []
    for doc in base_docs:
        result = compress_document(query, doc)
        if result is not None:
            compressed.append(result)

    logger.debug(
        "Contextual compression: %d -> %d docs for query: %.60s",
        len(base_docs), len(compressed), query,
    )
    return compressed

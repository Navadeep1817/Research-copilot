"""
researcher.py — Researcher Agent Node.

RESPONSIBILITY:
  Takes the current sub-question (from state.current_question_idx).
  Selects and executes the appropriate retrieval strategy.
  Passes retrieved docs through cross-encoder reranking.
  Generates a focused answer using retrieved context.
  Updates the sub_question with its answer + sources.
  Advances current_question_idx for the next iteration.

RETRIEVAL STRATEGY ROUTING:
  The researcher looks at state["retrieval_strategy"] and routes to
  the appropriate retrieval module. In the multi-agent design, the
  planner could also set per-sub-question strategies.

MULTI-HOP RESEARCH:
  This node runs in a LOOP (via LangGraph edges) until all
  sub-questions are answered. Each iteration:
    - picks sub_questions[current_question_idx]
    - retrieves evidence
    - generates answer
    - advances index

INTERVIEW: "The researcher is the core RAG node. It's stateless as a
function — all context comes from the state dict. The retrieval strategy
is configurable per-session, not hardcoded. I use a strategy router
pattern so adding a new retrieval method is adding one elif branch."

COMMON MISTAKE: Putting all retrieved documents into the LLM context
without reranking — the context window fills with low-relevance content
and faithfulness drops sharply.

ANSWER GENERATION PROMPT:
  - Instructs the LLM to answer ONLY from provided context
  - Asks for citation of source titles
  - Temperature=0 for factual accuracy
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.config import get_settings
from backend.agents.state import ResearchState
from backend.models.schemas import ResearchStatus, SourceDocument

# Retrieval imports
from backend.retrieval.dense_retriever import dense_search
from backend.retrieval.bm25_retriever import bm25_search
from backend.retrieval.hybrid_retriever import hybrid_search
from backend.retrieval.query_rewriter import rewrite_query
from backend.retrieval.multi_query import multi_query_search
from backend.retrieval.hyde import hyde_search
from backend.retrieval.parent_child import parent_child_search
from backend.retrieval.contextual_compression import contextual_compression_search
from backend.retrieval.metadata_filter import metadata_filter_search
from backend.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant. Answer the question using ONLY
the provided context documents. Be precise and factual.

If the context doesn't contain enough information, say:
"Insufficient information in retrieved sources to fully answer this question."

Always cite which sources support your claims using [Source: title] format."""),
    ("human", """Question: {question}

Context documents:
{context}

Provide a comprehensive answer based on the context above:"""),
])


def _route_retrieval(strategy: str, query: str, top_k: int) -> list[SourceDocument]:
    """Route to the appropriate retrieval strategy."""
    strategy = strategy.lower()

    if strategy == "dense":
        return dense_search(query, top_k=top_k)
    elif strategy == "bm25":
        return bm25_search(query, top_k=top_k)
    elif strategy == "hybrid":
        return hybrid_search(query, top_k=top_k)
    elif strategy == "query_rewrite":
        rewritten = rewrite_query(query)
        return hybrid_search(rewritten, top_k=top_k)
    elif strategy == "multi_query":
        return multi_query_search(query, top_k=top_k)
    elif strategy == "hyde":
        return hyde_search(query, top_k=top_k)
    elif strategy == "parent_child":
        return parent_child_search(query, top_k=top_k)
    elif strategy == "metadata_filter":
        return metadata_filter_search(query, {}, top_k=top_k)
    else:
        # Default to hybrid
        logger.warning("Unknown strategy '%s', defaulting to hybrid", strategy)
        return hybrid_search(query, top_k=top_k)


def researcher_node(state: ResearchState) -> dict:
    """
    LangGraph node: research one sub-question.

    Input state fields used:
        sub_questions, current_question_idx, retrieval_strategy
    Output state fields set:
        sub_questions (updated with answer+sources), sources, current_question_idx, status
    """
    settings = get_settings()
    idx = state.get("current_question_idx", 0)
    sub_questions = list(state["sub_questions"])  # copy to mutate

    if idx >= len(sub_questions):
        logger.info("All sub-questions answered, moving to critique")
        return {"status": ResearchStatus.CRITIQUE}

    current_q = sub_questions[idx]
    logger.info("Researcher working on Q%d: %s", idx + 1, current_q.question)

    # ── Step 1: Retrieve ──────────────────────────────────────────────────
    strategy = state.get("retrieval_strategy", "hybrid")
    raw_docs = _route_retrieval(strategy, current_q.question, top_k=settings.top_k_retrieval)

    # ── Step 2: Rerank ────────────────────────────────────────────────────
    reranked_docs = rerank(current_q.question, raw_docs, top_k=settings.top_k_rerank)

    # ── Step 3: Contextual compression (optional, on reranked top-K) ─────
    if reranked_docs:
        compressed = contextual_compression_search(current_q.question, reranked_docs)
        final_docs = compressed if compressed else reranked_docs
    else:
        final_docs = []

    # ── Step 4: Generate answer from context ──────────────────────────────
    if final_docs:
        context_str = "\n\n---\n\n".join(
            f"[Source: {d.title or d.source or d.doc_id}]\n{d.content}"
            for d in final_docs
        )
        try:
            llm = ChatGroq(
                model=settings.groq_model_name,
                temperature=0.0,
                api_key=settings.groq_api_key,
            )
            chain = ANSWER_PROMPT | llm | StrOutputParser()
            answer = chain.invoke({
                "question": current_q.question,
                "context": context_str,
            })
        except Exception as e:
            logger.error("Answer generation failed: %s", e)
            answer = f"Error generating answer: {e}"
    else:
        answer = "No relevant documents found in the knowledge base for this question."
        logger.warning("No docs retrieved for: %s", current_q.question)

    # ── Step 5: Update state ──────────────────────────────────────────────
    current_q.answer = answer
    current_q.answered = True
    current_q.sources = final_docs
    sub_questions[idx] = current_q

    next_idx = idx + 1
    next_status = ResearchStatus.RESEARCH if next_idx < len(sub_questions) else ResearchStatus.CRITIQUE

    logger.info("Researcher completed Q%d, advancing to idx=%d", idx + 1, next_idx)

    return {
        "sub_questions": sub_questions,
        "sources": final_docs,
        "current_question_idx": next_idx,
        "status": next_status,
    }

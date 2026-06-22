"""
planner.py — Planner Agent Node.

RESPONSIBILITY:
  Receives the raw user query.
  Decomposes it into 3-5 focused sub-questions.
  Creates a research plan (ordered list of what to investigate).
  Sets sub_questions and research_plan in state.

WHY DECOMPOSE:
  Complex queries like "Compare transformer vs LSTM architectures for
  NLP and discuss when each is preferred" cannot be answered by a single
  retrieval. Breaking into sub-questions allows:
    1. Targeted retrieval per sub-question
    2. Parallel research (future: fan-out in LangGraph)
    3. Structured synthesis from multiple evidence streams
    4. Critic can evaluate each sub-question independently

INTERVIEW: "The planner implements query decomposition, which is the
key innovation in systems like OpenAI Deep Research. Instead of one
big retrieval, we do N targeted retrievals and synthesize the results.
This improves both recall (we don't miss topics) and precision (each
retrieval is focused)."

COMMON MISTAKE: Generating too many sub-questions (>7) — retrieval
latency multiplies linearly. 3-5 is the sweet spot.

PROMPT ENGINEERING:
  - Explicit JSON output format prevents parsing errors
  - "self-contained" instruction ensures each question can be researched independently
  - Temperature=0 for deterministic decomposition
"""

import json
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.config import get_settings
from backend.agents.state import ResearchState
from backend.models.schemas import SubQuestion, ResearchStatus

logger = logging.getLogger(__name__)

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research planner. Given a complex research query,
decompose it into 3-5 focused sub-questions that together will answer the main query.

Rules:
- Each sub-question must be self-contained (answerable independently)
- Sub-questions should cover different aspects of the main query
- Order them logically (foundational concepts first)
- Keep each sub-question under 20 words

Respond with ONLY valid JSON in this exact format:
{{
  "research_plan": "Brief description of the overall research strategy",
  "sub_questions": [
    "Sub-question 1",
    "Sub-question 2",
    "Sub-question 3"
  ]
}}"""),
    ("human", "Research query: {query}"),
])


def planner_node(state: ResearchState) -> dict:
    """
    LangGraph node: decompose the user query into sub-questions.

    Input state fields used:  query
    Output state fields set:  sub_questions, research_plan, status
    """
    logger.info("Planner starting for query: %.80s", state["query"])
    settings = get_settings()

    llm = ChatGroq(
        model=settings.groq_model_name,
        temperature=0.0,
        api_key=settings.groq_api_key,
    )

    try:
        chain = PLANNER_PROMPT | llm | StrOutputParser()
        raw = chain.invoke({"query": state["query"]})

        # Strip markdown code fences if present
        raw = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)

        sub_questions = [
            SubQuestion(question=q)
            for q in data.get("sub_questions", [])
        ]

        if not sub_questions:
            # Fallback: treat the original query as a single sub-question
            sub_questions = [SubQuestion(question=state["query"])]

        logger.info("Planner created %d sub-questions", len(sub_questions))

        return {
            "sub_questions": sub_questions,
            "research_plan": data.get("research_plan", ""),
            "current_question_idx": 0,
            "retry_count": 0,
            "sources": [],
            "status": ResearchStatus.RESEARCH,
        }

    except Exception as e:
        logger.error("Planner failed: %s", e)
        # Graceful degradation: single sub-question = original query
        return {
            "sub_questions": [SubQuestion(question=state["query"])],
            "research_plan": "Direct research on the original query.",
            "current_question_idx": 0,
            "retry_count": 0,
            "sources": [],
            "status": ResearchStatus.RESEARCH,
            "error": str(e),
        }

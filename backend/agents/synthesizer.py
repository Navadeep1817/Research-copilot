"""
synthesizer.py — Synthesizer Agent Node.

RESPONSIBILITY:
  Receives all sub-question answers + sources from state.
  Synthesizes a comprehensive, well-structured research report.
  Adds citations in [Source N] format.
  Sets final_report in state.

WHAT GOOD SYNTHESIS LOOKS LIKE:
  - Integrates insights from multiple sub-question answers
  - Doesn't just concatenate — identifies connections and contrasts
  - Includes an executive summary, main body, and conclusions
  - Every factual claim is cited
  - Identifies remaining open questions or limitations

CITATION STRATEGY:
  Sources are numbered [1], [2], etc.
  A reference list is appended at the end.
  This makes the report auditable — readers can verify every claim.

INTERVIEW: "The synthesizer is the RAG generation step, but elevated
to research-report level. The prompt is carefully structured: it gets
the original query, the research plan, each sub-answer, and the sources.
It's instructed to synthesize — not summarize — meaning it should find
connections between sub-answers that weren't explicit in any individual
answer."

COMMON MISTAKE: Giving the synthesizer all raw chunks instead of the
processed sub-question answers. The sub-question answers are already
filtered and distilled — the synthesizer needs concise, structured input.

REPORT STRUCTURE (enforced by prompt):
  ## Executive Summary
  ## [Section per major theme]
  ## Conclusions
  ## References
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.config import get_settings
from backend.agents.state import ResearchState
from backend.models.schemas import ResearchStatus

logger = logging.getLogger(__name__)

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research analyst. Synthesize the provided research
findings into a comprehensive, well-structured research report.

Requirements:
- Write in clear, professional prose
- Start with an Executive Summary (3-5 sentences)
- Organize findings into logical sections with ## headings
- Cite sources using [Source N] notation inline
- End with Conclusions and a numbered References list
- Identify any gaps or limitations in the research
- Total length: 500-1000 words

Do NOT just concatenate the sub-answers — synthesize them into a coherent narrative."""),
    ("human", """Original Research Query: {query}

Research Plan: {research_plan}

Research Findings:
{findings}

Available Sources:
{sources_list}

Write a comprehensive research report:"""),
])


def synthesizer_node(state: ResearchState) -> dict:
    """
    LangGraph node: synthesize all research into a final report.

    Input state fields used:  query, research_plan, sub_questions, sources
    Output state fields set:  final_report, status
    """
    settings = get_settings()
    logger.info("Synthesizer generating report for query: %.80s", state["query"])

    # ── Build findings summary ────────────────────────────────────────────
    findings_parts = []
    for i, sq in enumerate(state.get("sub_questions", []), 1):
        if sq.answered and sq.answer:
            findings_parts.append(f"Finding {i} ({sq.question}):\n{sq.answer}")

    findings = "\n\n".join(findings_parts) if findings_parts else "No findings available."

    # ── Build sources list ────────────────────────────────────────────────
    all_sources = state.get("sources", [])
    # Deduplicate by content hash
    seen: set[str] = set()
    unique_sources = []
    for s in all_sources:
        key = s.content[:100]
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    sources_list = "\n".join(
        f"[Source {i+1}] {s.title or s.source or s.doc_id} (score: {s.score:.3f})"
        for i, s in enumerate(unique_sources[:20])
    )

    # ── Generate report ───────────────────────────────────────────────────
    try:
        llm = ChatGroq(
            model=settings.groq_model_name,
            temperature=0.2,   # slight creativity for better prose
            api_key=settings.groq_api_key,
            max_tokens=2048,
        )
        chain = SYNTHESIZER_PROMPT | llm | StrOutputParser()
        report = chain.invoke({
            "query": state["query"],
            "research_plan": state.get("research_plan", ""),
            "findings": findings,
            "sources_list": sources_list,
        })
        logger.info("Synthesizer generated %d char report", len(report))

    except Exception as e:
        logger.error("Synthesizer LLM call failed: %s", e)
        # Fallback: stitch sub-answers together
        report = f"# Research Report\n\n**Query:** {state['query']}\n\n"
        for sq in state.get("sub_questions", []):
            if sq.answered:
                report += f"## {sq.question}\n{sq.answer}\n\n"
        report += f"\n\n*Note: Report generation encountered an error: {e}*"

    return {
        "final_report": report,
        "status": ResearchStatus.COMPLETE,
    }

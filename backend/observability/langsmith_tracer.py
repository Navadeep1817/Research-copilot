"""
langsmith_tracer.py — LangSmith tracing setup.

WHY LANGSMITH: Every LLM call, prompt, token count, and latency is
automatically traced to LangSmith when LANGCHAIN_TRACING_V2=true.
This gives you a searchable UI for debugging agent runs.

WHAT LANGSMITH CAPTURES (automatically, no extra code needed):
  - Full prompt sent to each LLM call
  - LLM response (including streaming chunks)
  - Token usage (input / output / total)
  - Latency per call
  - Agent step sequence with inputs/outputs
  - Errors with full stack traces

SETUP: Set env vars. LangChain auto-instruments when tracing is enabled.
No explicit callback registration needed for LangChain/LangGraph calls.

INTERVIEW: "LangSmith is the observability layer for LLM calls.
It's to LLM apps what Datadog is to microservices. I use it to debug
retrieval quality issues — I can see exactly which prompt produced a
poor answer and which documents were retrieved."

COMMON MISTAKE: Leaving LANGCHAIN_TRACING_V2=true in production with
sensitive user data — traces contain full prompts. Use project-level
data retention policies and PII filtering.
"""

import logging
import os
from backend.config import get_settings

logger = logging.getLogger(__name__)


def setup_langsmith() -> bool:
    """
    Configure LangSmith tracing via environment variables.
    LangChain reads these at import time, so we set them early in main.py.

    Returns True if tracing is enabled.
    """
    settings = get_settings()

    if not settings.langchain_tracing_v2:
        logger.info("LangSmith tracing disabled (LANGCHAIN_TRACING_V2=false)")
        return False

    if not settings.langchain_api_key:
        logger.warning("LangSmith tracing enabled but LANGCHAIN_API_KEY is empty")
        return False

    # LangChain reads these env vars directly
    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
    os.environ["LANGCHAIN_API_KEY"]     = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"]     = settings.langchain_project

    logger.info("LangSmith tracing enabled for project: %s", settings.langchain_project)
    return True

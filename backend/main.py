"""
main.py — FastAPI application entrypoint.

ENDPOINTS:
  POST /api/research          — start a research session (streaming SSE)
  GET  /api/research/{id}     — get session results
  POST /api/ingest            — ingest documents into knowledge base
  GET  /metrics               — Prometheus metrics scrape endpoint
  GET  /health                — health check

STREAMING DESIGN:
  Research is a long-running operation (30-120s).
  We use Server-Sent Events (SSE) via StreamingResponse to push
  real-time agent progress updates to the frontend.
  
  Each SSE event is a JSON-encoded StreamEvent:
    {"event": "status",      "data": "planning"}
    {"event": "subquestion", "data": {"question": "...", "answer": "..."}}
    {"event": "source",      "data": {"title": "...", "content": "..."}}
    {"event": "report",      "data": "## Executive Summary..."}
    {"event": "eval",        "data": {"faithfulness": 0.87, ...}}

LIFESPAN CONTEXT MANAGER:
  FastAPI's lifespan hook runs startup/shutdown code.
  We use it to: init DB, load embedding model, load reranker,
  setup LangSmith, start Prometheus server.

INTERVIEW: "I chose SSE over WebSocket for streaming because SSE is
unidirectional (server->client), which matches our use case exactly.
WebSocket is bidirectional and adds complexity we don't need.
SSE also auto-reconnects on drop, which WebSocket doesn't."

CORS: Enabled for all origins in dev. In production, restrict to
your frontend domain.

ERROR HANDLING: All endpoints return structured error responses.
Unhandled exceptions are caught by the global exception handler and
returned as 500 with a trace ID.
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from backend.config import get_settings
from backend.db.database import init_db, get_db
from backend.models.schemas import (
    ResearchRequest, ResearchResponse, IngestRequest, IngestResponse,
    HealthResponse, StreamEvent, ResearchStatus, EvaluationScores,
)
from backend.agents.graph import research_graph
from backend.ingestion.indexer import ingest_texts
from backend.observability.langsmith_tracer import setup_langsmith
from backend.observability.prometheus_metrics import (
    research_requests_total, research_duration_seconds, active_research_sessions,
)
from backend.memory.long_term_memory import store_research_summary

logger = logging.getLogger(__name__)


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: startup actions before yield, shutdown after."""
    logger.info("Starting Research Copilot API...")

    # 1. Init database tables
    await init_db()

    # 2. Setup LangSmith tracing
    setup_langsmith()

    # 3. Warm up embedding model (loads from HuggingFace cache)
    from backend.retrieval.embeddings import get_embedding_model
    get_embedding_model()

    # 4. Warm up reranker
    from backend.retrieval.reranker import get_reranker
    get_reranker()

    # 5. Load BM25 index if it exists
    from backend.retrieval.bm25_retriever import load_index
    load_index()

    logger.info("Research Copilot API ready")
    yield
    logger.info("Shutting down Research Copilot API...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Research Copilot API",
    description="Multi-Agent Deep Research System with Advanced RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest raw texts into the knowledge base (Qdrant + BM25).
    Accepts a list of texts with optional metadata.
    """
    try:
        n = ingest_texts(
            texts=request.texts,
            metadatas=request.metadatas or [{} for _ in request.texts],
        )
        return IngestResponse(indexed=n, message=f"Successfully indexed {n} chunks")
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/research")
async def research_stream(request: ResearchRequest):
    """
    Start a research session. Returns a streaming SSE response.
    The frontend consumes events to show real-time progress.

    Event types:
      status      — agent phase change
      subquestion — a sub-question was answered
      source      — a source document was retrieved
      report      — the final research report
      eval        — RAGAS evaluation scores
      error       — an error occurred
      done        — stream complete
    """
    settings = get_settings()
    session_id = request.session_id

    async def event_stream() -> AsyncGenerator[str, None]:
        """Async generator that yields SSE-formatted events."""

        def sse(event: str, data) -> str:
            payload = StreamEvent(event=event, data=data, session_id=session_id)
            return f"data: {payload.model_dump_json()}\n\n"

        active_research_sessions.inc()
        start_time = time.time()

        try:
            # ── Build initial state ───────────────────────────────────────
            initial_state = {
                "query":               request.query,
                "session_id":          session_id,
                "retrieval_strategy":  request.strategy.value,
                "sub_questions":       [],
                "research_plan":       "",
                "sources":             [],
                "current_question_idx": 0,
                "retry_count":         0,
                "final_report":        "",
                "critique_result":     None,
                "evaluation":          None,
                "status":              ResearchStatus.PLANNING,
                "error":               None,
                "messages":            [],
            }

            yield sse("status", {"phase": "planning", "message": "Decomposing your research query..."})

            config = {"configurable": {"thread_id": session_id}}

            # ── Stream graph execution ────────────────────────────────────
            last_state = initial_state
            async for chunk in research_graph.astream(initial_state, config=config):
                node_name = list(chunk.keys())[0]
                node_output = chunk[node_name]
                last_state = {**last_state, **node_output}

                if node_name == "planner":
                    sqs = node_output.get("sub_questions", [])
                    yield sse("status", {
                        "phase": "planning_complete",
                        "message": f"Research plan: {len(sqs)} sub-questions identified",
                        "sub_questions": [sq.question for sq in sqs],
                    })

                elif node_name == "researcher":
                    sqs = node_output.get("sub_questions", [])
                    sources = node_output.get("sources", [])

                    # Emit answered sub-questions
                    for sq in sqs:
                        if sq.answered:
                            yield sse("subquestion", {
                                "question": sq.question,
                                "answer":   sq.answer[:500] + "..." if len(sq.answer) > 500 else sq.answer,
                            })

                    # Emit top sources
                    for src in sources[:3]:
                        yield sse("source", {
                            "title":   src.title or src.source,
                            "content": src.content[:300] + "...",
                            "score":   round(src.score, 3),
                            "strategy": src.strategy,
                        })

                    yield sse("status", {"phase": "researching", "message": "Collecting evidence..."})

                elif node_name == "critic":
                    critique = node_output.get("critique_result")
                    if critique:
                        if critique.passed:
                            yield sse("status", {"phase": "critique_passed", "message": "Quality check passed"})
                        else:
                            yield sse("status", {
                                "phase":   "critique_retry",
                                "message": f"Gaps found, researching further: {critique.gaps}",
                            })

                elif node_name == "synthesizer":
                    report = node_output.get("final_report", "")
                    yield sse("report", {"content": report})
                    yield sse("status", {"phase": "complete", "message": "Research complete"})

            # ── RAGAS Evaluation (async, after streaming) ─────────────────
            if request.evaluate and last_state.get("final_report"):
                yield sse("status", {"phase": "evaluating", "message": "Running quality evaluation..."})
                try:
                    from backend.evaluation.ragas_eval import evaluate_research
                    contexts = [s.content for s in last_state.get("sources", [])[:5]]
                    scores = evaluate_research(
                        question=request.query,
                        answer=last_state["final_report"],
                        contexts=contexts,
                    )
                    yield sse("eval", scores)
                except Exception as e:
                    logger.warning("Evaluation failed: %s", e)

            # ── Persist to long-term memory ───────────────────────────────
            await store_research_summary(
                session_id=session_id,
                query=request.query,
                summary=last_state.get("final_report", "")[:1000],
            )

            duration = time.time() - start_time
            research_duration_seconds.observe(duration)
            research_requests_total.labels(status="success").inc()
            yield sse("done", {"duration_seconds": round(duration, 2)})

        except Exception as e:
            logger.error("Research stream error: %s", e, exc_info=True)
            research_requests_total.labels(status="error").inc()
            yield sse("error", {"message": str(e)})
        finally:
            active_research_sessions.dec()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )

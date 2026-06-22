"""
schemas.py — All Pydantic request/response models.

WHY: Pydantic v2 models give automatic validation, serialization, and
OpenAPI schema generation for free.  Every API boundary is typed.

INTERVIEW: "I define all data contracts in one schemas.py so the API,
agents, and frontend all speak the same language.  Pydantic v2 is 5-17x
faster than v1 due to Rust-based validation core."

COMMON MISTAKE: Using dict everywhere instead of typed models — you lose
IDE autocomplete, validation, and OpenAPI docs generation.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import uuid


# ── Enums ─────────────────────────────────────────────────────────────────────

class ResearchStatus(str, Enum):
    PENDING   = "pending"
    PLANNING  = "planning"
    RESEARCH  = "researching"
    CRITIQUE  = "critiquing"
    SYNTHESIS = "synthesizing"
    COMPLETE  = "complete"
    FAILED    = "failed"


class RetrievalStrategy(str, Enum):
    DENSE                = "dense"
    BM25                 = "bm25"
    HYBRID               = "hybrid"
    QUERY_REWRITE        = "query_rewrite"
    MULTI_QUERY          = "multi_query"
    HYDE                 = "hyde"
    PARENT_CHILD         = "parent_child"
    CONTEXTUAL_COMPRESS  = "contextual_compress"
    METADATA_FILTER      = "metadata_filter"


# ── Core domain models ────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    """A retrieved document chunk with metadata."""
    doc_id:    str  = Field(default_factory=lambda: str(uuid.uuid4()))
    content:   str
    source:    str  = ""          # file path or URL
    title:     str  = ""
    score:     float = 0.0        # retrieval / rerank score
    strategy:  str  = ""          # which retrieval strategy found this
    metadata:  dict[str, Any] = Field(default_factory=dict)


class SubQuestion(BaseModel):
    """One decomposed research sub-question from the Planner."""
    id:         str  = Field(default_factory=lambda: str(uuid.uuid4()))
    question:   str
    answered:   bool = False
    answer:     str  = ""
    sources:    list[SourceDocument] = Field(default_factory=list)


class CritiqueResult(BaseModel):
    passed:       bool
    feedback:     str  = ""
    gaps:         list[str] = Field(default_factory=list)
    retry_count:  int  = 0


class EvaluationScores(BaseModel):
    faithfulness:      float | None = None
    context_precision: float | None = None
    context_recall:    float | None = None
    answer_relevance:  float | None = None


# ── API Request / Response models ─────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query:        str  = Field(..., min_length=10, max_length=2000)
    session_id:   str  = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy:     RetrievalStrategy = RetrievalStrategy.HYBRID
    max_sources:  int  = Field(default=5, ge=1, le=20)
    evaluate:     bool = Field(default=True)


class ResearchResponse(BaseModel):
    session_id:    str
    query:         str
    status:        ResearchStatus
    sub_questions: list[SubQuestion]     = Field(default_factory=list)
    report:        str                   = ""
    sources:       list[SourceDocument]  = Field(default_factory=list)
    critique:      CritiqueResult | None = None
    evaluation:    EvaluationScores | None = None
    created_at:    datetime              = Field(default_factory=datetime.utcnow)
    duration_ms:   int                  = 0


class StreamEvent(BaseModel):
    """Server-sent event payload for WebSocket streaming."""
    event:   str           # "status" | "subquestion" | "source" | "report" | "eval" | "error"
    data:    Any
    session_id: str = ""


class IngestRequest(BaseModel):
    texts:    list[str]
    metadatas: list[dict[str, Any]] = Field(default_factory=list)
    source:   str = "manual"


class IngestResponse(BaseModel):
    indexed:  int
    message:  str


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"

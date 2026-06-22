"""
fix_paths.py
============
Run this once from inside your research_copilot/ folder.
It patches config.py, database.py, dense_retriever.py, and
bm25_retriever.py to use absolute paths so scripts work from
any working directory.

Usage:
    cd C:\\Users\\Navadeep\\Downloads\\research_copilot
    python fix_paths.py
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent   # research_copilot/


def write(rel_path: str, content: str) -> None:
    p = HERE / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    print(f"  ✓ {rel_path}")


# ── 1. config.py ─────────────────────────────────────────────────────────────
write("backend/config.py", '''\
"""
config.py — Single source of truth for all configuration.
Uses __file__ to find .env relative to project root (not cwd).
"""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py is at  research_copilot/backend/config.py
# PROJECT_ROOT  =  research_copilot/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE     = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    groq_api_key: str     = Field(..., description="Groq API key (required)")
    groq_model_name: str  = Field("llama-3.1-70b-versatile")

    # LangSmith
    langchain_tracing_v2: bool = Field(False)
    langchain_api_key: str     = Field("")
    langchain_project: str     = Field("research-copilot")

    # Qdrant
    qdrant_path: str              = Field("./data/qdrant_storage")
    qdrant_collection_name: str   = Field("research_docs")
    qdrant_parent_collection: str = Field("research_docs_parent")

    # Embeddings
    embedding_model: str  = Field("BAAI/bge-small-en-v1.5")
    embedding_device: str = Field("cpu")
    embedding_dim: int    = Field(384)

    # Reranker
    reranker_model: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Database
    database_url: str = Field("./data/research_copilot.db")

    # API
    api_host: str  = Field("0.0.0.0")
    api_port: int  = Field(8000)
    log_level: str = Field("INFO")

    # RAG
    chunk_size: int              = Field(512)
    chunk_overlap: int           = Field(64)
    top_k_retrieval: int         = Field(10)
    top_k_rerank: int            = Field(5)
    max_research_iterations: int = Field(3)

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def qdrant_storage_path(self) -> Path:
        p = Path(self.qdrant_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def db_path(self) -> Path:
        p = Path(self.database_url)
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton. Call get_settings.cache_clear() in tests."""
    return Settings()
''')


# ── 2. database.py ───────────────────────────────────────────────────────────
write("backend/db/database.py", '''\
"""database.py — Async SQLite via aiosqlite. Uses absolute db_path."""

import aiosqlite
import logging
from backend.config import get_settings

logger = logging.getLogger(__name__)

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT \'pending\',
    report      TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration_ms INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    content     TEXT NOT NULL,
    source      TEXT,
    title       TEXT,
    score       REAL DEFAULT 0.0,
    strategy    TEXT,
    metadata    TEXT DEFAULT \'{}\'
);
CREATE TABLE IF NOT EXISTS sub_questions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT,
    answered    INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS evaluations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL UNIQUE,
    faithfulness      REAL,
    context_precision REAL,
    context_recall    REAL,
    answer_relevance  REAL,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS long_term_memory (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT NOT NULL UNIQUE,
    value      TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sources_session ON sources(session_id);
CREATE INDEX IF NOT EXISTS idx_subq_session    ON sub_questions(session_id);
"""


async def get_db():
    settings = get_settings()
    db_path = settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()


async def init_db() -> None:
    settings = get_settings()
    db_path = settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(db_path)) as db:
        await db.executescript(CREATE_TABLES)
        await db.commit()
    logger.info("Database initialised at %s", db_path)
''')


# ── 3. dense_retriever.py ────────────────────────────────────────────────────
write("backend/retrieval/dense_retriever.py", '''\
"""dense_retriever.py — Qdrant ANN vector search. Uses absolute storage path."""

import logging
import uuid
from typing import Any
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from backend.config import get_settings
from backend.models.schemas import SourceDocument

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    path = settings.qdrant_storage_path      # absolute Path
    path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(path))
    logger.info("Qdrant client at %s", path)
    return client


def ensure_collection(collection_name: str | None = None) -> None:
    settings = get_settings()
    name = collection_name or settings.qdrant_collection_name
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s", name)


def upsert_documents(
    texts: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str] | None = None,
    collection_name: str | None = None,
) -> int:
    from backend.retrieval.embeddings import embed_documents
    settings = get_settings()
    name = collection_name or settings.qdrant_collection_name
    ensure_collection(name)
    client = get_qdrant_client()

    vectors = embed_documents(texts)
    points = [
        PointStruct(
            id=ids[i] if ids else str(uuid.uuid4()),
            vector=vectors[i],
            payload={"content": texts[i], **(metadatas[i] if i < len(metadatas) else {})},
        )
        for i in range(len(texts))
    ]
    client.upsert(collection_name=name, points=points)
    logger.info("Upserted %d points into %s", len(points), name)
    return len(points)


def dense_search(
    query: str,
    top_k: int | None = None,
    collection_name: str | None = None,
    filter_conditions: dict[str, Any] | None = None,
) -> list[SourceDocument]:
    from backend.retrieval.embeddings import embed_query
    settings = get_settings()
    k = top_k or settings.top_k_retrieval
    name = collection_name or settings.qdrant_collection_name
    client = get_qdrant_client()

    # Check collection exists
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        logger.warning("Collection %s does not exist yet — returning empty results", name)
        return []

    query_vector = embed_query(query)

    qdrant_filter = None
    if filter_conditions:
        qdrant_filter = Filter(
            must=[FieldCondition(key=k, match=MatchValue(value=v))
                  for k, v in filter_conditions.items()]
        )

    results = client.search(
        collection_name=name,
        query_vector=query_vector,
        limit=k,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    docs = []
    for hit in results:
        payload = hit.payload or {}
        docs.append(SourceDocument(
            doc_id=str(hit.id),
            content=payload.get("content", ""),
            source=payload.get("source", ""),
            title=payload.get("title", ""),
            score=hit.score,
            strategy="dense",
            metadata={k: v for k, v in payload.items() if k not in ("content", "source", "title")},
        ))
    return docs
''')


# ── 4. bm25_retriever.py — fix absolute cache path ───────────────────────────
write("backend/retrieval/bm25_retriever.py", '''\
"""bm25_retriever.py — BM25 sparse retrieval. Cache stored under project data/."""

import logging
import pickle
from pathlib import Path
from typing import Any
from backend.config import get_settings
from backend.models.schemas import SourceDocument

logger = logging.getLogger(__name__)


def _bm25_cache_path() -> Path:
    settings = get_settings()
    return settings.data_dir / "bm25_index.pkl"


def tokenize(text: str) -> list[str]:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    return nltk.word_tokenize(text.lower())


_corpus_texts: list[str] = []
_corpus_metadatas: list[dict[str, Any]] = []
_bm25_index = None


def build_index(texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> None:
    global _corpus_texts, _corpus_metadatas, _bm25_index
    from rank_bm25 import BM25Okapi
    _corpus_texts     = texts
    _corpus_metadatas = metadatas or [{} for _ in texts]
    tokenised   = [tokenize(t) for t in texts]
    _bm25_index = BM25Okapi(tokenised)
    cache = _bm25_cache_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump({"texts": _corpus_texts, "metadatas": _corpus_metadatas, "index": _bm25_index}, f)
    logger.info("BM25 index built with %d documents", len(texts))


def load_index() -> bool:
    global _corpus_texts, _corpus_metadatas, _bm25_index
    cache = _bm25_cache_path()
    if not cache.exists():
        return False
    with open(cache, "rb") as f:
        data = pickle.load(f)
    _corpus_texts     = data["texts"]
    _corpus_metadatas = data["metadatas"]
    _bm25_index       = data["index"]
    logger.info("BM25 index loaded (%d docs)", len(_corpus_texts))
    return True


def add_documents(texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> None:
    global _corpus_texts, _corpus_metadatas
    _corpus_texts.extend(texts)
    _corpus_metadatas.extend(metadatas or [{} for _ in texts])
    build_index(_corpus_texts, _corpus_metadatas)


def bm25_search(query: str, top_k: int | None = None) -> list[SourceDocument]:
    global _bm25_index, _corpus_texts, _corpus_metadatas
    settings = get_settings()
    k = top_k or settings.top_k_retrieval
    if _bm25_index is None:
        if not load_index():
            logger.warning("BM25 index not built yet")
            return []
    tokens = tokenize(query)
    scores = _bm25_index.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    max_score = scores[top_indices[0]] if top_indices and scores[top_indices[0]] > 0 else 1.0
    docs = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        meta = _corpus_metadatas[idx] if idx < len(_corpus_metadatas) else {}
        docs.append(SourceDocument(
            content=_corpus_texts[idx],
            source=meta.get("source", ""),
            title=meta.get("title", ""),
            score=float(scores[idx]) / max_score,
            strategy="bm25",
            metadata=meta,
        ))
    return docs
''')


# ── 5. long_term_memory.py — fix absolute db path ────────────────────────────
write("backend/memory/long_term_memory.py", '''\
"""long_term_memory.py — SQLite-backed persistence. Uses absolute db path."""

import json
import logging
from datetime import datetime
import aiosqlite
from backend.config import get_settings

logger = logging.getLogger(__name__)


async def store_memory(key: str, value: str | dict) -> None:
    settings = get_settings()
    val = json.dumps(value) if isinstance(value, dict) else value
    async with aiosqlite.connect(str(settings.db_path)) as db:
        await db.execute(
            """INSERT INTO long_term_memory (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, val, datetime.utcnow().isoformat()),
        )
        await db.commit()


async def retrieve_memory(key: str) -> str | None:
    settings = get_settings()
    async with aiosqlite.connect(str(settings.db_path)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT value FROM long_term_memory WHERE key=?", (key,)) as cur:
            row = await cur.fetchone()
    return row["value"] if row else None


async def list_memory_keys(prefix: str = "") -> list[str]:
    settings = get_settings()
    async with aiosqlite.connect(str(settings.db_path)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT key FROM long_term_memory WHERE key LIKE ?", (f"{prefix}%",)
        ) as cur:
            rows = await cur.fetchall()
    return [r["key"] for r in rows]


async def store_research_summary(session_id: str, query: str, summary: str) -> None:
    await store_memory(f"research:{session_id}", json.dumps({"query": query, "summary": summary}))
''')


print("\n✅ All path fixes applied successfully!")
print("\nNow run:")
print("  python scripts/seed_data.py")
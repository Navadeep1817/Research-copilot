"""database.py — Async SQLite via aiosqlite. Uses absolute db_path."""

import aiosqlite
import logging
from backend.config import get_settings

logger = logging.getLogger(__name__)

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
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
    metadata    TEXT DEFAULT '{}'
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

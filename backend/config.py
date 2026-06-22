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

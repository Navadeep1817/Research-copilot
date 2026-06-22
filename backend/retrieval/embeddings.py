"""
embeddings.py — BGE embedding wrapper (singleton, thread-safe).

WHY BGE: BAAI/bge-small-en-v1.5 outperforms OpenAI ada-002 on MTEB
benchmarks while running entirely on CPU.  bge-large-en-v1.5 is
available as a drop-in upgrade (change EMBEDDING_MODEL in .env).

WHY SINGLETON: SentenceTransformer loads a 100-500MB model from disk.
Loading it on every request would make the API unusable.  We load once
at startup and share across all requests.

INTERVIEW: "I wrapped the embedding model in a module-level singleton
so it loads once and is reused across threads. BGE models prepend a
query instruction for asymmetric retrieval — 'Represent this sentence
for searching relevant passages:' for queries, nothing for documents."

COMMON MISTAKE: Not prepending the BGE instruction prefix to queries.
BGE models are trained with this prefix and perform 3-5% worse without it.

PERFORMANCE: batch_size=32 on CPU gives ~50 sentences/sec on bge-small.
For ingestion of large corpora, increase batch_size to 128+.

ALTERNATIVES: OpenAI text-embedding-3-small (better quality, API cost),
Cohere embed-v3 (multilingual), local all-MiniLM-L6-v2 (smaller/faster).
"""

import logging
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.config import get_settings

logger = logging.getLogger(__name__)

# BGE query instruction prefix (critical for retrieval quality)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model (called once at startup)."""
    settings = get_settings()
    logger.info("Loading embedding model: %s on %s", settings.embedding_model, settings.embedding_device)
    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    logger.info("Embedding model loaded. Dim=%d", model.get_sentence_embedding_dimension())
    return model


def embed_query(query: str) -> list[float]:
    """
    Embed a search query.
    Prepends the BGE instruction prefix for asymmetric retrieval.
    Returns a normalised float list ready for Qdrant cosine search.
    """
    model = get_embedding_model()
    prefixed = BGE_QUERY_PREFIX + query
    vector = model.encode(prefixed, normalize_embeddings=True)
    return vector.tolist()


def embed_documents(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Embed a list of document chunks for indexing.
    Documents do NOT get the query prefix — this is the asymmetric design.
    Returns list of normalised float vectors.
    """
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )
    return [v.tolist() for v in vectors]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two normalised vectors."""
    va = np.array(a)
    vb = np.array(b)
    return float(np.dot(va, vb))

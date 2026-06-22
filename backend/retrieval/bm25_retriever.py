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

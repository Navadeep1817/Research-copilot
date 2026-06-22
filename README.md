# 🔬 Research Copilot — Deep Research AI System

A production-grade Multi-Agent AI Research Copilot with Advanced RAG.
Modelled after OpenAI Deep Research, Perplexity Deep Research, and Gemini Deep Research.

---

## Architecture

```
Streamlit Frontend
      ↓ SSE streaming
FastAPI Backend
      ↓
LangGraph StateGraph
  Planner → Researcher (loop) → Critic → Synthesizer
      ↓
Advanced RAG Layer (9 strategies + cross-encoder reranking)
  Dense · BM25 · Hybrid RRF · QueryRewrite · MultiQuery
  HyDE · ParentChild · CtxCompression · MetadataFilter
      ↓
Storage: Qdrant (local) + SQLite + BM25 in-memory
      ↓
Observability: LangSmith + Prometheus + RAGAS
```

---

## Quick Start

### 1. Setup virtual environment

```cmd
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```cmd
pip install -r requirements.txt
```

> For CPU-only PyTorch (avoids CUDA errors on Windows):
> ```cmd
> pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 3. Configure environment

```cmd
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux
```

Edit `.env` and fill in:
- `GROQ_API_KEY` — free at https://console.groq.com
- `LANGCHAIN_API_KEY` — free at https://smith.langchain.com (optional)

### 4. Seed the knowledge base

```cmd
python scripts/seed_data.py
```

### 5. Start the backend

```cmd
cd research_copilot
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Start the frontend (new terminal)

```cmd
cd research_copilot
streamlit run frontend/app.py
```

Open http://localhost:8501

---

## Run Tests

```cmd
pytest tests/ -v
```

---

## Ingest Your Own Documents

```cmd
# Upload via API
python scripts/ingest_docs.py --dir data/raw

# Or use the Streamlit sidebar file uploader
```

---

## Project Structure

```
research_copilot/
├── backend/
│   ├── main.py                    # FastAPI app + SSE streaming
│   ├── config.py                  # Pydantic Settings
│   ├── agents/
│   │   ├── graph.py               # LangGraph StateGraph
│   │   ├── state.py               # ResearchState TypedDict
│   │   ├── planner.py             # Query decomposition
│   │   ├── researcher.py          # RAG retrieval + answer generation
│   │   ├── critic.py              # Quality evaluation + retry
│   │   └── synthesizer.py         # Report generation
│   ├── retrieval/
│   │   ├── embeddings.py          # BGE embeddings
│   │   ├── dense_retriever.py     # Qdrant ANN search
│   │   ├── bm25_retriever.py      # BM25 sparse search
│   │   ├── hybrid_retriever.py    # RRF fusion
│   │   ├── query_rewriter.py      # LLM query rewriting
│   │   ├── multi_query.py         # Parallel query variants
│   │   ├── hyde.py                # Hypothetical Document Embedding
│   │   ├── parent_child.py        # Small-to-big retrieval
│   │   ├── contextual_compression.py  # LLM extraction
│   │   ├── metadata_filter.py     # Qdrant payload filtering
│   │   └── reranker.py            # Cross-encoder reranking
│   ├── memory/
│   │   ├── conversation_memory.py # Sliding window buffer
│   │   ├── research_memory.py     # Per-session evidence store
│   │   └── long_term_memory.py    # SQLite persistence
│   ├── evaluation/
│   │   └── ragas_eval.py          # RAGAS metrics
│   ├── ingestion/
│   │   ├── document_loader.py     # File loading (txt/pdf/html/md)
│   │   ├── chunker.py             # Recursive + parent-child chunking
│   │   └── indexer.py             # Full ingestion pipeline
│   ├── observability/
│   │   ├── langsmith_tracer.py    # LangSmith setup
│   │   └── prometheus_metrics.py  # Metrics registry
│   ├── models/
│   │   └── schemas.py             # Pydantic models
│   └── db/
│       └── database.py            # SQLite async setup
├── frontend/
│   └── app.py                     # Streamlit UI
├── scripts/
│   ├── seed_data.py               # Seed sample documents
│   └── ingest_docs.py             # CLI ingestion tool
├── tests/
│   ├── test_retrieval.py
│   ├── test_agents.py
│   └── test_evaluation.py
├── data/
│   ├── raw/                       # Drop documents here
│   └── qdrant_storage/            # Qdrant local persistence
├── .env.example
└── requirements.txt
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph |
| LLM | Groq API (Llama 3.1 70B) |
| Vector DB | Qdrant (local) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Sparse Retrieval | BM25 (rank-bm25) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Backend | FastAPI + uvicorn |
| Frontend | Streamlit |
| Evaluation | RAGAS |
| Tracing | LangSmith |
| Metrics | Prometheus |
| Database | SQLite (aiosqlite) |

---

## Interview Talking Points

- **Why LangGraph?** Cyclic workflows (critic→researcher retry loop) cannot be expressed in linear LCEL chains
- **Why RRF?** Dense and BM25 scores are incomparable ranges; RRF fuses by rank position (score-agnostic)
- **Why BGE + cross-encoder?** Two-stage: bi-encoder for fast ANN retrieval, cross-encoder for precise reranking
- **Why HyDE?** Short queries embed poorly vs dense documents; hypothetical doc bridges the distribution gap
- **Why RAGAS?** 4 orthogonal metrics diagnose retrieval vs generation failures independently

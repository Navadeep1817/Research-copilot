"""
prometheus_metrics.py — Prometheus metrics registry.

METRICS WE TRACK:
  Counters:   research_requests_total, retrieval_calls_total
  Histograms: research_duration_seconds, llm_call_duration_seconds,
              retrieval_duration_seconds, rerank_duration_seconds
  Gauges:     active_research_sessions, qdrant_collection_size

WHY PROMETHEUS + GRAFANA:
  - De facto standard for microservice monitoring
  - Pull-based: Prometheus scrapes /metrics endpoint
  - Histograms give P50/P95/P99 latency percentiles (not just averages)
  - Grafana visualises time-series for dashboards and alerting

INTERVIEW: "I expose a /metrics endpoint in FastAPI using
prometheus-client. Prometheus scrapes it every 15s and stores time-series.
I track histograms (not averages) because P99 latency is what users
actually experience during worst-case requests."

COMMON MISTAKE: Using a Counter for latency (Counters only go up).
Use Histogram or Summary for latency measurements.

LABEL STRATEGY: Keep label cardinality low. Don't label by session_id
(millions of series). Label by strategy, model, status instead.
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY

# ── Counters ──────────────────────────────────────────────────────────────────
research_requests_total = Counter(
    "research_requests_total",
    "Total number of research requests",
    ["status"],   # labels: success, error
)

retrieval_calls_total = Counter(
    "retrieval_calls_total",
    "Total retrieval calls by strategy",
    ["strategy"],
)

llm_calls_total = Counter(
    "llm_calls_total",
    "Total LLM API calls by agent",
    ["agent"],    # planner, researcher, critic, synthesizer
)

# ── Histograms ────────────────────────────────────────────────────────────────
research_duration_seconds = Histogram(
    "research_duration_seconds",
    "End-to-end research request duration",
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

llm_call_duration_seconds = Histogram(
    "llm_call_duration_seconds",
    "LLM API call duration by agent",
    ["agent"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

retrieval_duration_seconds = Histogram(
    "retrieval_duration_seconds",
    "Retrieval latency by strategy",
    ["strategy"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
)

rerank_duration_seconds = Histogram(
    "rerank_duration_seconds",
    "Cross-encoder reranking duration",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0],
)

# ── Gauges ────────────────────────────────────────────────────────────────────
active_research_sessions = Gauge(
    "active_research_sessions",
    "Number of currently active research sessions",
)

retrieval_results_count = Histogram(
    "retrieval_results_count",
    "Number of documents returned per retrieval call",
    ["strategy"],
    buckets=[0, 1, 2, 5, 10, 20],
)

"""
app.py — Streamlit frontend for Research Copilot.
Fixed: ingest metadatas format, port consistency, better error messages.
"""

import json
import uuid
import httpx
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Research Copilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")   # change to 8080 if you started uvicorn on 8080

defaults = {
    "current_report": "",
    "current_sources": [],
    "eval_scores": {},
    "sub_questions": [],
    "workflow_log": [],
    "research_done": False,
    "last_query": "",
    "error_msg": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Copilot")
    st.markdown("*Deep Research AI System*")
    st.divider()

    st.subheader("⚙️ Configuration")
    strategy    = st.selectbox("Retrieval Strategy",
                               ["hybrid","dense","bm25","multi_query","hyde","query_rewrite","parent_child"])
    max_sources = st.slider("Max Sources per Query", 3, 15, 5)
    run_eval    = st.checkbox("Run Evaluation", value=True)

    st.divider()
    st.subheader("📚 Ingest Documents")
    uploaded_file = st.file_uploader("Upload a text document", type=["txt","md"])

    if uploaded_file:
        if st.button("📥 Ingest Document"):
            try:
                # Read file content
                raw_bytes = uploaded_file.read()
                text = raw_bytes.decode("utf-8", errors="replace")

                if not text.strip():
                    st.error("File appears to be empty.")
                else:
                    filename = uploaded_file.name
                    title    = filename.rsplit(".", 1)[0]   # strip extension

                    with st.spinner(f"Indexing {filename}... ({len(text):,} chars)"):
                        resp = httpx.post(
                            f"{API_BASE}/api/ingest",
                            json={
                                "texts": [text],
                                "metadatas": [{          # ← correct format
                                    "source": filename,
                                    "title":  title,
                                }],
                            },
                            timeout=120,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            n    = data.get("indexed", 0)
                            st.success(f"✅ Indexed {n} chunks from **{filename}**")
                            st.caption(f"Now ask a question about {title}!")
                        else:
                            st.error(f"Server error {resp.status_code}: {resp.text[:200]}")

            except httpx.ConnectError:
                st.error(f"❌ Cannot reach backend at {API_BASE}. Is uvicorn running?")
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    st.divider()
    if st.button("🗑️ Clear Session"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

    st.divider()
    st.caption("LangGraph · Qdrant · BGE · Groq")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 Deep Research Copilot")
st.markdown(
    "Ask a complex research question. The AI decomposes it, retrieves evidence, "
    "critiques findings, and synthesizes a cited report."
)

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_area(
        "Query",
        placeholder="e.g. How does RAG compare to fine-tuning for domain adaptation in LLMs?",
        height=80,
        label_visibility="collapsed",
    )
with col2:
    st.write("")
    research_btn = st.button("🚀 Research", use_container_width=True, type="primary")


# ── Run research ──────────────────────────────────────────────────────────────
if research_btn and query.strip():
    for k, v in defaults.items():
        st.session_state[k] = v
    st.session_state.last_query = query

    session_id = str(uuid.uuid4())
    progress_placeholder = st.empty()

    with progress_placeholder.container():
        st.info("⏳ Research in progress... (30–90 seconds)")
        prog    = st.progress(0)
        log_area = st.empty()

    log_lines: list[str]  = []
    sources_collected     = []
    sub_questions_collected = []
    report_text  = ""
    eval_data    = {}
    error_text   = ""

    try:
        with httpx.Client(timeout=300) as client:
            with client.stream(
                "POST",
                f"{API_BASE}/api/research",
                json={
                    "query":       query,
                    "session_id":  session_id,
                    "strategy":    strategy,
                    "max_sources": max_sources,
                    "evaluate":    run_eval,
                },
            ) as response:
                pct = 5
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        payload = json.loads(line[5:].strip())
                    except Exception:
                        continue

                    event = payload.get("event", "")
                    data  = payload.get("data", {})

                    if event == "status":
                        phase   = data.get("phase", "")
                        message = data.get("message", "")
                        log_lines.append(f"✅ **{phase}** — {message}")
                        pct = {
                            "planning": 10, "planning_complete": 20,
                            "researching": 50, "critique_passed": 70,
                            "critique_retry": 55, "synthesizing": 85,
                            "evaluating": 95, "complete": 100,
                        }.get(phase, pct)
                        prog.progress(pct)
                        log_area.markdown("\n\n".join(log_lines[-6:]))
                        if "sub_questions" in data:
                            for sq in data["sub_questions"]:
                                sub_questions_collected.append({"question": sq, "answer": ""})

                    elif event == "subquestion":
                        q = data.get("question", "")
                        a = data.get("answer", "")
                        log_lines.append(f"💡 Answered: *{q[:70]}*")
                        log_area.markdown("\n\n".join(log_lines[-6:]))
                        for sq in sub_questions_collected:
                            if sq["question"] == q:
                                sq["answer"] = a

                    elif event == "source":
                        sources_collected.append(data)

                    elif event == "report":
                        report_text = data.get("content", "")
                        log_lines.append("📋 **Report generated!**")
                        log_area.markdown("\n\n".join(log_lines[-6:]))

                    elif event == "eval":
                        eval_data = data
                        log_lines.append("📊 **Evaluation complete!**")
                        log_area.markdown("\n\n".join(log_lines[-6:]))

                    elif event == "error":
                        error_text = data.get("message", "Unknown error")
                        log_lines.append(f"❌ Error: {error_text[:100]}")
                        log_area.markdown("\n\n".join(log_lines[-6:]))

                    elif event == "done":
                        prog.progress(100)
                        break

    except httpx.ConnectError:
        error_text = f"❌ Cannot connect to backend at {API_BASE}. Is uvicorn running?"
    except Exception as e:
        error_text = str(e)

    st.session_state.current_report    = report_text
    st.session_state.current_sources   = sources_collected
    st.session_state.eval_scores       = eval_data
    st.session_state.sub_questions     = sub_questions_collected
    st.session_state.workflow_log      = log_lines
    st.session_state.error_msg         = error_text
    st.session_state.research_done     = True

    progress_placeholder.empty()
    st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.research_done:

    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)

    # Report
    if st.session_state.current_report:
        st.divider()
        st.subheader("📋 Research Report")
        st.caption(f"Query: *{st.session_state.last_query}*")
        with st.container(border=True):
            st.markdown(st.session_state.current_report)
        st.download_button(
            "⬇️ Download Report as Markdown",
            data=st.session_state.current_report,
            file_name="research_report.md",
            mime="text/markdown",
        )
    else:
        if not st.session_state.error_msg:
            st.warning("⚠️ No report was generated. Check the backend terminal for errors.")

    # Evaluation
    if st.session_state.eval_scores:
        st.divider()
        st.subheader("📊 Quality Evaluation (RAGAS)")
        scores = st.session_state.eval_scores
        c1, c2, c3, c4 = st.columns(4)
        metric_data = [
            (c1, "🎯 Faithfulness",      scores.get("faithfulness"),      "Grounded in sources?"),
            (c2, "💬 Answer Relevance",  scores.get("answer_relevance"),  "On-topic answer?"),
            (c3, "🔍 Context Precision", scores.get("context_precision"), "Low retrieval noise?"),
            (c4, "📚 Context Recall",    scores.get("context_recall"),    "Full coverage?"),
        ]
        for col, name, val, help_txt in metric_data:
            with col:
                if val is not None:
                    st.metric(name, f"{val:.2f}", help=help_txt)
                    st.progress(float(val))
                else:
                    st.metric(name, "N/A", help=help_txt)

        valid = [(n.split(" ", 1)[1], v) for _, n, v, _ in metric_data if v is not None]
        if len(valid) >= 3:
            names  = [n for n, _ in valid]
            values = [v for _, v in valid]
            fig = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=names + [names[0]],
                fill="toself",
                fillcolor="rgba(255,75,75,0.2)",
                line_color="#FF4B4B",
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False, height=280,
                margin=dict(l=40, r=40, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Sources
    if st.session_state.current_sources:
        st.divider()
        st.subheader(f"📄 Retrieved Sources ({len(st.session_state.current_sources)})")
        for i, src in enumerate(st.session_state.current_sources[:10], 1):
            title   = src.get("title") or src.get("source") or f"Source {i}"
            score   = src.get("score", 0)
            strat   = src.get("strategy", "")
            content = src.get("content", "")[:300]
            with st.expander(f"**{i}. {title}** — score: {score:.3f} · {strat}"):
                st.markdown(content + "...")

    # Sub-questions
    if st.session_state.sub_questions:
        st.divider()
        st.subheader(f"🔍 Sub-Questions ({len(st.session_state.sub_questions)})")
        for i, sq in enumerate(st.session_state.sub_questions, 1):
            with st.expander(f"Q{i}: {sq['question']}"):
                st.markdown(sq.get("answer") or "*No answer recorded*")

    # Workflow log
    if st.session_state.workflow_log:
        st.divider()
        with st.expander("🤖 Agent Workflow Log"):
            for entry in st.session_state.workflow_log:
                st.markdown(entry)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"🔬 Research Copilot | Backend: {API_BASE} | "
    f"[API Docs]({API_BASE}/docs) · [Metrics]({API_BASE}/metrics) · [Health]({API_BASE}/health)"
)
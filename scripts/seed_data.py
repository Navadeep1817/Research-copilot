"""
seed_data.py — Seed the knowledge base with sample research documents.

Run this ONCE after setup to have content to research against.
Adds 10 high-quality AI/ML passages covering RAG, LLMs, and agents.

Usage:
    cd research_copilot
    python scripts/seed_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ingestion.indexer import ingest_texts

SAMPLE_DOCUMENTS = [
    {
        "text": """Retrieval Augmented Generation (RAG) is a technique that combines 
large language models with external knowledge retrieval. Instead of relying solely 
on parametric knowledge (learned during training), RAG systems retrieve relevant 
documents at inference time and use them as context for generation. This approach 
reduces hallucinations, allows the model to access up-to-date information, and 
provides citations for generated content. RAG was introduced by Lewis et al. in 2020 
and has become the dominant paradigm for knowledge-intensive NLP tasks. The retrieval 
component typically uses dense vector search (bi-encoders like DPR or BGE), sparse 
search (BM25), or hybrid combinations of both. The generation component takes the 
retrieved passages as additional context alongside the user query.""",
        "meta": {"title": "Introduction to RAG", "source": "ai_overview", "year": 2024},
    },
    {
        "text": """Large Language Models (LLMs) have demonstrated remarkable capabilities 
across a wide range of NLP tasks. Models like GPT-4, Claude, and Llama 3 are trained 
on trillions of tokens using transformer architecture with self-attention mechanisms. 
The scaling laws discovered by Kaplan et al. show that model performance improves 
predictably with compute, data, and parameters. Fine-tuning techniques like RLHF 
(Reinforcement Learning from Human Feedback) and DPO (Direct Preference Optimization) 
align these models with human preferences. However, LLMs still suffer from 
hallucinations — confidently stating false information — which RAG and other 
grounding techniques aim to mitigate. Context window sizes have grown from 2K tokens 
(GPT-2) to 1M+ tokens (Gemini 1.5 Pro), enabling new long-document reasoning tasks.""",
        "meta": {"title": "Large Language Models Overview", "source": "ai_overview", "year": 2024},
    },
    {
        "text": """Vector databases are specialized storage systems optimized for 
high-dimensional vector similarity search. Unlike traditional databases that use 
B-tree indexes for exact matching, vector databases use Approximate Nearest Neighbor 
(ANN) algorithms like HNSW (Hierarchical Navigable Small World), IVF (Inverted File 
Index), and ScaNN (Scalable Nearest Neighbors). Qdrant, Pinecone, Weaviate, and 
Chroma are leading vector database solutions. HNSW builds a multi-layer graph where 
the top layer has long-range connections and lower layers have short-range connections, 
enabling logarithmic search time. Vector databases support payload filtering, allowing 
structured metadata conditions (date ranges, categories) to be combined with semantic 
search. This is essential for production RAG systems where freshness and source 
filtering are required.""",
        "meta": {"title": "Vector Databases for AI", "source": "ai_overview", "year": 2024},
    },
    {
        "text": """Agentic AI systems are autonomous software agents that use LLMs as 
their reasoning engine to plan and execute multi-step tasks. Unlike simple chatbots, 
agents can use tools (web search, code execution, APIs), maintain memory across 
interactions, and adapt their strategy based on feedback. The ReAct (Reasoning and 
Acting) framework interleaves chain-of-thought reasoning with action execution, 
allowing agents to observe results and adjust plans dynamically. LangGraph provides 
a graph-based framework for building stateful multi-agent systems with explicit 
control flow, cycles for self-correction, and human-in-the-loop capabilities. 
Common agent architectures include: Tool-using agents (function calling), 
Multi-agent systems (specialized sub-agents), and Planning agents (task decomposition 
followed by parallel execution). Key challenges include: tool reliability, error 
propagation in long chains, and cost management.""",
        "meta": {"title": "Agentic AI Systems", "source": "ai_overview", "year": 2024},
    },
    {
        "text": """Transformer architecture, introduced by Vaswani et al. in 2017 in 
'Attention Is All You Need', revolutionized natural language processing. The key 
innovation is the self-attention mechanism, which allows each token to attend to all 
other tokens in the sequence. Multi-head attention runs attention in parallel across 
H different learned subspaces, capturing different types of relationships. The 
feed-forward network after each attention layer applies position-wise transformations. 
Positional encodings (sinusoidal or learned) inject sequence order information since 
attention is permutation-invariant. Layer normalization stabilizes training. The 
encoder-decoder architecture suits sequence-to-sequence tasks; decoder-only 
transformers (GPT-style) suit autoregressive generation; encoder-only (BERT-style) 
suit classification and embedding tasks. Transformer has O(n²) attention complexity 
in sequence length n, which motivated efficient attention variants like FlashAttention, 
Sparse Attention, and linear attention methods.""",
        "meta": {"title": "Transformer Architecture", "source": "ml_fundamentals", "year": 2024},
    },
    {
        "text": """Fine-tuning vs RAG is a fundamental architectural choice when 
adapting LLMs to specific domains. Fine-tuning updates model weights by training 
on domain-specific data, permanently encoding knowledge into parameters. RAG keeps 
model weights frozen and retrieves knowledge at inference time. Trade-offs: Fine-tuning 
provides faster inference (no retrieval latency) and can teach the model new skills 
(reasoning styles, output formats) but requires expensive training compute, risks 
catastrophic forgetting of general knowledge, and cannot update facts without 
retraining. RAG provides updatable knowledge (just re-index new documents), better 
fact attribution (sources are explicit), and lower training cost, but adds retrieval 
latency and depends on retrieval quality. In practice, the best systems combine both: 
fine-tune for style and reasoning, use RAG for factual grounding. LoRA (Low-Rank 
Adaptation) makes fine-tuning more accessible by training only low-rank adapter 
matrices (typically <1% of parameters) while keeping the base model frozen.""",
        "meta": {"title": "Fine-tuning vs RAG", "source": "ai_overview", "year": 2024},
    },
    {
        "text": """Evaluation of RAG systems requires metrics that capture both 
retrieval quality and generation quality. RAGAS (Retrieval Augmented Generation 
Assessment) provides four key metrics: (1) Faithfulness measures whether the 
generated answer is factually consistent with the retrieved context — calculated 
by decomposing the answer into atomic claims and verifying each against the context. 
(2) Answer Relevance measures whether the answer addresses the question — calculated 
by generating alternative questions from the answer and measuring their similarity 
to the original. (3) Context Precision measures what fraction of retrieved chunks 
were relevant to answering the question. (4) Context Recall measures whether all 
facts needed to answer the question were present in the retrieved context. 
Traditional NLP metrics like BLEU and ROUGE are insufficient for RAG because they 
require reference answers and cannot detect hallucinations. LLM-as-judge approaches 
use a stronger LLM (GPT-4) to evaluate answers, providing more nuanced quality 
assessment but at higher cost.""",
        "meta": {"title": "RAG Evaluation Metrics", "source": "ai_overview", "year": 2024},
    },
    {
        "text": """Embedding models convert text into dense vector representations 
that capture semantic meaning. Sentence transformers use BERT-style encoders with 
mean pooling over token embeddings to produce fixed-size sentence vectors. BAAI 
(Beijing Academy of AI) BGE models are among the strongest open-source embedding 
models, achieving SOTA results on MTEB (Massive Text Embedding Benchmark). 
BGE-small-en-v1.5 (33M parameters, dim=384) runs efficiently on CPU while 
BGE-large-en-v1.5 (335M parameters, dim=1024) provides higher quality at 10x 
compute cost. The training objective for retrieval-optimized embeddings uses 
contrastive learning: positive (query, relevant_doc) pairs are pulled together 
in embedding space while negative (query, random_doc) pairs are pushed apart. 
Hard negative mining — using retrieved but non-relevant documents as negatives — 
dramatically improves retrieval quality compared to random negatives. For 
asymmetric retrieval (short query vs long document), models like BGE use 
instruction prefixes to differentiate query embeddings from document embeddings.""",
        "meta": {"title": "Text Embedding Models", "source": "ml_fundamentals", "year": 2024},
    },
    {
        "text": """Prompt engineering encompasses techniques for eliciting better 
outputs from LLMs without changing model weights. Key techniques include: 
Chain-of-thought (CoT) prompting — adding 'Let's think step by step' or 
providing reasoning examples improves multi-step reasoning accuracy by up to 40%. 
Few-shot prompting provides input-output examples in the prompt to demonstrate 
the desired format and reasoning style. ReAct combines reasoning traces with 
action execution in an interleaved fashion. Tree of Thoughts (ToT) explores 
multiple reasoning paths and uses evaluation to select the best continuation. 
Self-consistency generates multiple solutions and selects by majority vote. 
Structured output prompting requests JSON/XML format to enable reliable parsing. 
Temperature controls output randomness (0=deterministic, 1=creative). 
System prompts establish the AI's persona, constraints, and context. 
Prompt injection attacks occur when user content overwrites system instructions — 
mitigation requires input sanitization and instruction hierarchy.""",
        "meta": {"title": "Prompt Engineering Techniques", "source": "ai_overview", "year": 2024},
    },
    {
        "text": """LangGraph is a framework for building stateful, multi-actor 
applications with LLMs. It extends LangChain with a graph-based execution model 
that supports cycles, branching, and parallel execution — features impossible in 
linear LCEL chains. Core concepts: StateGraph defines a typed state dict shared 
across all nodes. Nodes are Python functions that receive the current state and 
return a partial state update. Edges connect nodes and can be conditional — a 
function inspects the state and returns the next node name. MemorySaver provides 
checkpointing so long-running agent workflows can be persisted and resumed. 
The compile() method validates the graph (checks for unreachable nodes, missing 
edges) and returns an executable with invoke(), stream(), and astream() methods. 
Streaming yields partial state updates after each node execution, enabling 
real-time progress display in frontends. LangGraph is used in production by 
companies like Elastic, Klarna, and LinkedIn for complex agent orchestration.""",
        "meta": {"title": "LangGraph Framework", "source": "ai_overview", "year": 2024},
    },
]


def main():
    print("Seeding knowledge base with sample AI/ML documents...")
    texts = [d["text"] for d in SAMPLE_DOCUMENTS]
    metas = [d["meta"] for d in SAMPLE_DOCUMENTS]

    n = ingest_texts(texts, metas)
    print(f"✅ Seeded {n} chunks from {len(SAMPLE_DOCUMENTS)} documents")
    print("\nYou can now research topics like:")
    print("  - How does RAG compare to fine-tuning for domain adaptation?")
    print("  - Explain the transformer attention mechanism and its complexity")
    print("  - What evaluation metrics should I use for a RAG system?")


if __name__ == "__main__":
    main()

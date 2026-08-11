# AI QA Lab — LLM Evaluation Harness for FinTech

Hybrid evaluation harness for a RAG assistant over a **credit policy** domain and a **real regulator document** (CBR Regulation 590-P, 150+ pages).
Fully local (Ollama), no API keys, no data leakage — suitable for regulated environments (banking, fintech).

## Key results

- **Real regulator RAG:** 96% deterministic pass rate on a 25-case Golden Dataset (answer / refuse / multi-hop over tables) over full-text CBR 590-P.
- **Eval-driven retrieval debugging on real text:** dense-only 44% → hybrid v1 60% → hybrid v2 96% (OR-semantics FTS + curated table/clause propositions).
- **Grounding + honest refusal:** the assistant cites source facts and correctly refuses when data is absent.
- **Robust judge circuit:** DeepEval LLM-as-a-Judge with a local Ollama judge (JSON-mode, temperature=0, graceful degradation via `safe_measure`).
- Insight documented: a *correct refusal* scores 0.5 on Answer Relevancy — metrics encode product priorities, not universal truth.

## Architecture

- `first_llm_call.py` — Step 1: first HTTP call to Ollama + latency/token-speed metrics.
- `eval_primitive.py` — Step 2: 3 heuristic checks (completeness, script purity, language purity).
- `eval_batch.py` — Step 3: batch runner, pass_rate, failure distribution, JSON report.
- `rag_assistant.py` — RAG over 590-P: chunking → nomic-embed-text → pgvector → **hybrid retrieval** (vector + tsvector, RRF fusion) → generation.
- `rag_eval.py` — DeepEval + local LLM-as-a-Judge (Faithfulness, Answer Relevancy).
- `rag_eval_batch.py` — batch harness over the Golden Dataset (deterministic layer + optional judge).
- `golden_dataset_590p.py` — 25 cases: direct facts, refusals, multi-hop (Table 1 → Table 2).
- `tables_590p.py` — table serialization: Tables 1/2 propositions, Table 1⋈2 join propositions, clause propositions.
- `load_590p.py` — PDF → text parser for the regulator document.
- `add_fts.py` — migration: generated `tsvector` column (Russian stemmer) + GIN index.
- `add_table_chunks.py` — idempotent incremental indexing of propositions.
- `debug_retrieval.py` — top-k retrieval diagnostics (the eval-driven debugging tool).
- `db_check.py`, `vector_roundtrip.py` — infrastructure smoke tests.
- `docker-compose.yml`, `init.sql` — PostgreSQL 16 + pgvector in Docker (healthcheck, persistent volume, auto-init).
- `documents/doc_590-P.txt` — full text of CBR Regulation 590-P.
- `results/` — reference reports (the eval-driven history):
  - `01_baseline_40pct.json`, `02_rag_first_87pct.json`, `03_final_100pct.json` (synthetic credit policy)
  - `04_590p_dense_only_44pct.json`, `05_590p_hybrid_v1_60pct.json`, `06_590p_hybrid_v2_96pct.json` (real regulator)

## Stack

- **LLM inference:** Ollama (qwen2.5)
- **Embeddings:** Ollama (nomic-embed-text, 768-d)
- **Vector store:** PostgreSQL 16 + pgvector (HNSW, cosine) in Docker Compose
- **Lexical search:** tsvector with Russian stemmer + GIN index; RRF fusion with the vector branch
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`
- **LLM-as-a-Judge:** DeepEval (Faithfulness, Answer Relevancy)
- **Python:** requests, psycopg2-binary, pdfplumber

## Observability (opt-in)

- LangSmith tracing via `@traceable` spans: `embedding.nomic-embed-text`, `generation.qwen2.5`, root `rag.pipeline`.
- Disabled by default (fully-local mode preserved); enabled via `.env` (template: `.env.example`):
  `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`.
- OS env vars take precedence over `.env` — CI secrets override the local file.
- Measured on CPU: cold start ~136s, warm 22–56s per request (generation-bound) — the baseline for capacity planning.

## Prerequisites

- Python 3.11+, Docker Desktop, Ollama
- On Windows with NVIDIA GPU + old CUDA drivers, force CPU inference via user environment variables:


CUDA_VISIBLE_DEVICES=-1
OLLAMA_VULKAN=0
OLLAMA_LLM_LIBRARY=cpu_avx2


## Setup

```bash
# 1. Pull models (one-time)
ollama pull qwen2.5
ollama pull nomic-embed-text

# 2. Start PostgreSQL + pgvector
docker compose up -d

# 3. Python environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 4. Migrations + index
python add_fts.py            # tsvector column + GIN index
python add_table_chunks.py   # incremental proposition indexing


Running

python rag_assistant.py       # standalone RAG demo over 590-P (builds index if empty)
python rag_eval_batch.py      # Golden Dataset batch eval -> results/rag_batch_590p_*.json
python rag_eval.py            # DeepEval judge demo (slow, ~10 min)
python debug_retrieval.py     # retrieval diagnostics (top-k + similarity)


Results

Synthetic credit policy (methodology bootstrap)
 Iteration 1 — Baseline (heuristics only, no RAG): pass rate 40%. LLM hallucinations, code-switching, script mixing. The model answered from parametric memory without grounding.
 Iteration 2 — After RAG (first run): pass rate 87.5%. Grounding worked; one retrieval miss on dividend question. Localized via JSON report as a retrieval error, not a generation error.
 Iteration 3 — After retrieval fix: pass rate 100%. Applied k=3 and persisted retrieval context in the report for reproducibility. Eval-driven fix confirmed by reference report.

Real regulator document (CBR 590-P, 150+ pages)
 04 — dense-only 44%: embeddings alone cannot discriminate homogeneous legal text; tables destroyed by PDF→text conversion; paraphrased queries fail exact-match retrieval.
 05 — hybrid v1 60%: vector + FTS added, but plainto_tsquery uses AND-semantics by default — paraphrased queries silently dropped the relevant chunks. Bug localized via debug_retrieval.py.
 06 — hybrid v2 96%: OR-semantics FTS (replacing & with | in the tsquery) + serialized tables/joins/clauses in the index. One residual FAIL on a near-duplicate proposition case — documented in the roadmap.
Reference JSON reports are committed in results/ so the full diagnostic path is reproducible.

Eval-driven debugging: how each iteration was diagnosed
Every jump in pass rate came from a hypothesis → fix → measurement cycle:
 44% → 60%: debug_retrieval.py showed the relevant table chunks were not in the top-3. Fix: add table serialization as standalone propositions.
 60% → 96%: inspection of the remaining FAILs revealed that FTS queries built by plainto_tsquery used AND — queries phrased differently from the source text had no FTS match at all. Fix: rewrite the query with OR semantics and rank by ts_rank.
 One residual FAIL: near-duplicate propositions (e.g., "среднее+среднее" vs "среднее+хорошее") confuse ranking — a known limitation that requires a cross-encoder reranker.
This is the same loop used in production AI QA: localize the failing layer (retrieval vs generation vs judge), apply a minimal fix, re-measure, commit the report.


Roadmap
 Cross-encoder reranker for near-duplicate propositions (last FAIL in 06)
 RAGAS (Context Precision / Context Recall) as a second metric family
 LangGraph multi-agent: retriever → generator → validator
 pandas analysis of eval reports (pass_rate evolution charts, failure distribution by question type)
 GitLab CI: auto-run eval on push, publish JSON report as artifact
 Custom metric: CorrectnessOfRefusal (refuse = PASS when data is absent, explicitly coded as product priority)

Why this domain
The credit-policy domain is not a toy example — it's my 13.5-year banking background (treasury, transfer pricing, covenant monitoring, regulatory reporting). The Golden Dataset encodes real banking requirements (reserve categories, classification matrices, reporting deadlines from CBR 590-P), not generic QA pairs.
Working with 590-P specifically demonstrates handling of:
Regulator-grade documents: 150+ pages, formal legal language, cross-references between chapters.
Tabular knowledge: Tables 1 and 2 of 590-P encode the entire loan-classification logic as matrices; naive RAG fails on them.
Grounding constraints: in banking, every answer must be traceable to a specific clause; hallucinations are not a UX issue, they are a compliance issue.
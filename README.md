# AI QA Lab — LLM Evaluation Harness for FinTech

Hybrid evaluation harness for a RAG assistant over a **credit policy** domain (covenants, Debt/EBITDA, DSCR).
Fully local (Ollama), no API keys, no data leakage — suitable for regulated environments (banking, fintech).

## Key results

- RAG assistant with **grounding**: cites source facts (e.g. 3.0x) and **correctly refuses** when data is missing.
- **Eval-driven fix**: localized a retrieval error via JSON report (pass rate **87.5% → 100%**).
- **Robust judge circuit**: JSON-mode + temperature=0 + graceful degradation (`safe_measure`) for LLM-as-a-Judge.
- Insight documented: a *correct refusal* scores 0.5 on `Answer Relevancy` — metrics encode product priorities, not universal truth.


## Architecture

- `first_llm_call.py` — Step 1: first HTTP call to Ollama + latency/token-speed metrics.
- `eval_primitive.py` — Step 2: 3 heuristic checks (completeness, script purity, language purity).
- `eval_batch.py` — Step 3: batch runner, pass_rate, failure distribution, JSON report.
- `rag_assistant.py` — Step 4: full RAG (chunking → embeddings → ChromaDB → retrieval → generation).
- `rag_eval.py` — Step 5: DeepEval + local LLM-as-a-Judge (Faithfulness, Answer Relevancy).
- `rag_eval_batch.py` — Step 6: Golden Dataset (8 cases, answer/refuse) + hybrid eval.
- `requirements.txt` — direct dependencies, grouped by layer.
- `results/` — reference reports (commit history of the eval-driven cycle):
    - `01_baseline_40pct.json`
    - `02_rag_first_87pct.json`
    - `03_final_100pct.json`



## Stack

- **LLM inference:** Ollama (qwen2.5)
- **Embeddings:** Ollama (nomic-embed-text)
- **Vector store:** ChromaDB (in-memory for demo)
- **RAG:** LangChain (`RecursiveCharacterTextSplitter`)
- **LLM-as-a-Judge:** DeepEval (Faithfulness, Answer Relevancy)
- **HTTP client:** `requests`

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running
- On Windows with NVIDIA GPU + old CUDA drivers, force CPU inference via user environment variables:


CUDA_VISIBLE_DEVICES=-1
OLLAMA_VULKAN=0
OLLAMA_LLM_LIBRARY=cpu_avx2


## Setup

```bash
# 1. Pull models (one-time)
ollama pull qwen2.5
ollama pull nomic-embed-text

# 2. Python environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# 3. Dependencies
pip install -r requirements.txt


Running
Start Ollama server first (if not running as a background service):
bash
ollama serve


Then pick a script:
# Quick smoke test: first LLM call + latency
python first_llm_call.py

# Heuristic eval (completeness / script purity / language purity)
python eval_primitive.py

# Batch run: pass_rate + failure distribution
python eval_batch.py

# RAG assistant (standalone demo)
python rag_assistant.py

# DeepEval + local LLM-as-a-Judge
python rag_eval.py

# Golden Dataset (8 cases, answer/refuse) + hybrid eval
python rag_eval_batch.py


Results
Iteration 1 — Baseline (heuristics only, no RAG)
Pass rate: 40%.
LLM hallucinations, code-switching, script mixing. The model answered from parametric memory without grounding.

Iteration 2 — After RAG (first run)
Pass rate: 87.5%.
Grounding worked; one retrieval miss on dividend question. Localized via JSON report as a retrieval error, not a generation error.

Iteration 3 — After retrieval fix
Pass rate: 100%.
Applied k=3 and persisted retrieval context in the report for reproducibility. Eval-driven fix confirmed by reference report.
Reference JSON reports are committed in results/ so the full diagnostic path is reproducible.

Roadmap
Swap ChromaDB → pgvector (PostgreSQL) — aligns with DWH expertise
Realistic 100+ page credit policy with tables (covenant thresholds by rating)
Multi-hop questions (e.g. "Debt/EBITDA for BB-rated borrower AND breach consequences")
GitLab CI: auto-run eval on push, publish JSON report as artifact
Custom metric: CorrectnessOfRefusal (refuse = PASS when data is absent)
Streaming evaluation + TTFT / tokens-per-sec SLAs
# Self-Healing Document RAG Agent — Acme Corp HR Assistant

A staged, end-to-end RAG agent for HR policy Q&A, built with LangChain/
LangGraph, running entirely on local open-weight models via Ollama (no paid
API keys required). Built in four stages, each adding one production
concern on top of a working baseline:

| Stage | What it adds |
|---|---|
| 0 | FastAPI + Streamlit RAG MVP — tool-calling agent over Qdrant (vector search) + DuckDB (structured employee lookup) |
| 1 | LangGraph self-healing loop — retrieve → generate → critique → retry, so an ungrounded answer gets caught and retried instead of shipped |
| 2 | Guardrails-as-config — YAML-driven input (prompt-injection/PII) and output (banned-topic) filtering, no code changes needed to adjust policy |
| 3 | Golden-dataset eval harness + GitHub Actions CI gate — 19 test cases, tool-accuracy + keyword-coverage scoring, fails the build below threshold |
| 4 (optional) | Swappable LLM provider (Ollama ↔ vLLM/OpenAI-compatible) and an optional Neo4j knowledge-graph tool |

## Why local models

No budget for paid LLM APIs during development, so every stage runs on
open-weight models served locally by [Ollama](https://ollama.com) —
`qwen2.5:7b` for chat, `nomic-embed-text` for embeddings. `qwen2.5:1.5b` was
tried first and is noticeably faster, but testing showed it reliably
skipped tool calls on real questions (answering PTO and other policy
questions generically instead of searching the actual documents) — `7b`
was the model that fixed that specific, reproduced failure. That failure
mode is also the direct motivation for Stage 1's critique/retry loop:
trusting a single LLM pass unconditionally is fragile, so Stage 1 adds a
second pass that checks the first one's work.

## Setup

**1. Install Ollama and pull the models**

```bash
brew install ollama          # or download from https://ollama.com/download
ollama serve &                # if it isn't already running as a background service
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

**2. Python environment**

```bash
cd policy-agent
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

**3. Ingest the sample documents** (already included under `data/`; only
regenerate if you want fresh synthetic data)

```bash
python3 scripts/generate_sample_data.py   # optional
python3 -c "from app.ingest import run_ingestion; run_ingestion()"
```

## Testing each stage

```bash
python3 scripts/smoke_test.py          # Stage 0 — simple tool-calling loop
python3 scripts/smoke_test_stage1.py   # Stage 1 — self-healing graph
python3 scripts/smoke_test_stage2.py   # Stage 2 — guardrails (no LLM needed, instant)
python3 scripts/eval_stage3.py         # Stage 3 — full 19-case golden-dataset eval
```

`eval_stage3.py` writes a full `eval_report.json` with every question,
answer, and pass/fail reason — useful for actually reading what the model
said, not just the aggregate score.

## Running the app

```bash
uvicorn app.main:app --reload --port 8000     # terminal 1
streamlit run app/streamlit_app.py            # terminal 2
```

Open the Streamlit URL (usually `http://localhost:8501`). The sidebar lets
you switch between Stage 0 (simple loop) and Stage 1 (self-healing) per
question, and answers show a grounded/retry badge (Stage 1) or a guardrail
block indicator (Stage 2) when relevant.

## CI (Stage 3)

`.github/workflows/eval.yml` runs the full eval harness on every push/PR to
`main`: installs Ollama, pulls both models, ingests the sample data, runs
`scripts/eval_stage3.py`, and fails the build if the pass rate drops below
75% (`EVAL_PASS_THRESHOLD`). GitHub-hosted runners have no GPU, so this can
take 10-20+ minutes on `qwen2.5:7b` — `actions/cache` caches the pulled
models between runs so only the first run pays the full download cost.

## Guardrails (Stage 2)

`guardrails.yaml` holds plain regex/keyword rules — prompt-injection
phrases and PII patterns on the input side, banned topics on the output
side — kept out of code entirely so a policy change is a YAML edit, not a
deploy. See `app/guardrails.py`.

## Stage 4 (optional)

Both pieces are disabled by default so the rest of the app is unaffected:

- **Swappable LLM provider** (`app/llm.py`) — set `LLM_PROVIDER=vllm` plus
  `VLLM_BASE_URL`/`VLLM_MODEL` in `.env` to point every chat call at a
  self-hosted, OpenAI-compatible [vLLM](https://docs.vllm.ai) endpoint
  serving a bigger open-weight model instead of Ollama. Not run against a
  live vLLM deployment in this project (no GPU available during
  development) — written and syntax-checked, not yet execution-verified.
- **Neo4j knowledge-graph tool** (`app/graph_tool.py`,
  `scripts/ingest_graph.py`) — set `NEO4J_ENABLED=true` plus your Neo4j
  connection details (e.g. from [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/)'s
  free tier) to extract entities/relationships from the policy PDFs via
  LangChain's `LLMGraphTransformer` and add a `search_relationships` tool
  for connection-style questions. Same caveat: written defensively (every
  failure mode degrades to a plain-text message instead of crashing the
  app) but not run against a live Neo4j instance.

## Known limitations

- **Stage 3's keyword scoring is a cheap faithfulness proxy, not an LLM
  judge.** It checks whether expected keywords/phrases appear anywhere in
  the answer, which is fast and needs no extra model call, but it can
  false-positive: a decline like *"I don't have information about parental
  leave"* contains the word "parental," which could accidentally satisfy a
  badly-written keyword check even though the answer said nothing
  substantive. Worth tightening with an LLM-as-judge scorer (e.g. Ragas) if
  this were taken further.
- **Stage 4's vLLM and Neo4j paths are untested against live
  infrastructure** — no GPU or hosted graph DB was available during
  development. Both are written defensively and syntax-verified, but treat
  them as a starting point to validate against your own instance, not
  pre-verified code.
- **`QdrantClient(path=...)` local mode locks `qdrant_data/` for exclusive
  access** — only one process (ingestion, smoke tests, or `uvicorn`) can
  have it open at a time. Stop one before starting another. Swapping to a
  real Qdrant server removes this limitation.

## Tech stack

Python, FastAPI, Streamlit, LangChain, LangGraph, Ollama (`qwen2.5:7b`,
`nomic-embed-text`), Qdrant (vector store), DuckDB (structured lookups),
PyMuPDF (PDF parsing), PyYAML (guardrails config), GitHub Actions (CI eval
gate), and optionally vLLM and Neo4j for Stage 4.

## Project layout

```
policy-agent/
  data/
    policies/*.pdf           synthetic sample HR policy documents
    employees.csv             synthetic sample employee roster
  app/
    config.py                  all settings, env-var overridable
    ingest.py                   PDF -> chunks -> embeddings -> Qdrant
    db.py                        DuckDB connection over employees.csv
    tools.py                      search_policies / search_employee / list_documents
    agent.py                       Stage 0: simple tool-calling loop
    graph.py                        Stage 1: LangGraph self-healing loop
    guardrails.py                    Stage 2: input/output guardrail checks
    llm.py                            Stage 4: swappable chat-LLM provider factory
    graph_tool.py                      Stage 4 (optional): Neo4j relationship tool
    main.py                              FastAPI server (POST /chat, stage-selectable)
    streamlit_app.py                      chat UI
  eval/
    golden_dataset.json          Stage 3: 19-case eval set
  scripts/
    generate_sample_data.py      (re)generate the sample PDFs + CSV
    smoke_test.py                 Stage 0 acceptance test
    smoke_test_stage1.py           Stage 1 acceptance test
    smoke_test_stage2.py            Stage 2 guardrail test
    eval_stage3.py                   Stage 3 eval harness (also runs in CI)
    ingest_graph.py                   Stage 4 (optional): builds the Neo4j graph
  .github/workflows/
    eval.yml                      Stage 3 CI gate
  guardrails.yaml                Stage 2 policy config
  requirements.txt
  .env.example
```

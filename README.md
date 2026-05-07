# FinSight AI

A multi-agent financial intelligence system powered by Claude. Answers questions from SEC filings, earnings call transcripts, and market reports using RAG, a research agent, and a memory agent — all routed through an LLM orchestrator.

## Architecture

```
User Query
    │
    ▼
Orchestrator  ──► route decision (LLM)
    │
    ├──► RAGAgent        — retrieves from ChromaDB vector store
    ├──► ResearchAgent   — deep analysis with tool use
    └──► MemoryAgent     — recalls past session context
    │
    ▼
FastAPI MCP Server  ◄──► Streamlit Dashboard

LLM Provider layer (providers/)
    ├── GeminiProvider    — PRIMARY  (GEMINI_API_KEY)
    └── AnthropicProvider — FALLBACK (ANTHROPIC_API_KEY, optional)
         └── LLMRouter retries fallback on any ProviderError (auth, billing, quota)

Resilience layers (outermost to innermost):
  1. LLMRouter           — retries secondary provider on primary failure
  2. Per-agent fallback  — RAG returns raw retrieved documents; Research/Memory
                           return a clear "service unavailable" message
  3. API endpoint        — returns 503 (not 500) if ProviderError escapes agents
```

## Quickstart

```bash
# 1. Clone and install
git clone <repo>
cd finsight-ai
python -m venv .venv && source .venv/bin/activate
make install

# 2. Configure environment
cp .env.example .env
# Required: set GEMINI_API_KEY  (primary provider)
# Optional: set ANTHROPIC_API_KEY  (fallback — enables automatic failover)

# 3. Ingest sample documents
make ingest

# 4. Start the API server (terminal 1)
make dev

# 5. Start the dashboard (terminal 2)
make dashboard
```

Open [http://localhost:8501](http://localhost:8501) to use the Streamlit interface.

## Docker

```bash
cp .env.example .env   # fill in GEMINI_API_KEY (and optionally ANTHROPIC_API_KEY)
make docker-up
```

Services:
- API: [http://localhost:8000](http://localhost:8000)
- Dashboard: [http://localhost:8501](http://localhost:8501)
- PostgreSQL: `localhost:5432`

## Evaluation

```bash
make test
```

The `evaluation/` suite covers RAG retrieval accuracy, agent routing logic, and LLM output quality. Tests use mocked LLM clients and run without real API keys.

## Resilience and Provider Fallback

| Failure scenario | Behaviour |
|---|---|
| Primary provider (Gemini) auth/billing error | LLMRouter retries with Anthropic fallback (if `ANTHROPIC_API_KEY` is set) |
| Only Anthropic configured, Anthropic fails | RAG agent returns raw retrieved document excerpts; Research/Memory return a service-unavailable message |
| No documents ingested and all providers down | Clear message asking the user to run `make ingest` and check provider keys |
| Truly unexpected error (bug) | API returns 500 with `"Internal server error."` — details in server logs |

The API schema is unchanged: every `/api/v1/query` call returns a `{"answer": ..., "route": ...}` JSON body regardless of provider status.

## Project Layout

```
agents/         — Orchestrator, RAGAgent, ResearchAgent, MemoryAgent
providers/      — LLM provider abstraction (Gemini primary, Anthropic fallback)
mcp_server/     — FastAPI server and API routes
evaluation/     — pytest suite
data/           — ingestion script and sample financial filings
dashboard/      — Streamlit UI
db/             — PostgreSQL schema
assets/demo/    — screenshots and demo assets
```

## Environment Variables

See `.env.example` for all supported variables.

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | **Required** (primary). Your Google Gemini API key. |
| `GEMINI_MODEL` | Gemini model name (default: `gemini-2.0-flash`) |
| `ANTHROPIC_API_KEY` | Optional (fallback). Your Anthropic API key. Enables automatic failover when Gemini is unavailable or returns an auth/billing error. |
| `ANTHROPIC_MODEL` | Anthropic model name (default: `claude-sonnet-4-6`) |
| `FINSIGHT_API_URL` | Dashboard → API URL (default: `http://localhost:8000/api/v1`) |
| `DATABASE_URL` | PostgreSQL connection string (optional) |
| `CHROMA_PERSIST_DIR` | ChromaDB persistence path (default: `./data/chroma_db`) |

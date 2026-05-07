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
# Edit .env — set ANTHROPIC_API_KEY

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
cp .env.example .env   # fill in ANTHROPIC_API_KEY
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

The `evaluation/` suite covers RAG retrieval accuracy, agent routing logic, and LLM output quality using mocked Anthropic clients.

## Project Layout

```
agents/         — Orchestrator, RAGAgent, ResearchAgent, MemoryAgent
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
| `ANTHROPIC_API_KEY` | Required. Your Anthropic API key. |
| `FINSIGHT_API_URL` | Dashboard → API URL (default: `http://localhost:8000/api/v1`) |
| `DATABASE_URL` | PostgreSQL connection string (optional) |
| `CHROMA_PERSIST_DIR` | ChromaDB persistence path (default: `./data/chroma_db`) |

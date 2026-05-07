from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI

from .api import router
from agents import Orchestrator
from providers import build_llm_router

logger = logging.getLogger(__name__)


def create_server() -> FastAPI:
    app = FastAPI(title="FinSight AI", version="0.1.0", description="Financial intelligence MCP server")
    app.include_router(router, prefix="/api/v1")

    try:
        llm_router = build_llm_router()
    except (RuntimeError, ImportError) as exc:
        logger.critical("Failed to initialise LLM providers: %s", exc)
        raise

    orchestrator = Orchestrator(llm_router)
    app.state.orchestrator = orchestrator

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    uvicorn.run(create_server(), host="0.0.0.0", port=8000)

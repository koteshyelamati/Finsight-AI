from __future__ import annotations

import os

import anthropic
import uvicorn
from fastapi import FastAPI

from .api import router
from agents import Orchestrator


def create_server() -> FastAPI:
    app = FastAPI(title="FinSight AI", version="0.1.0", description="Financial intelligence MCP server")
    app.include_router(router, prefix="/api/v1")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    orchestrator = Orchestrator(client)
    app.state.orchestrator = orchestrator

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    uvicorn.run(create_server(), host="0.0.0.0", port=8000)

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from providers.base import ProviderError

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    route: str
    session_id: str | None = None


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest, request: Request) -> QueryResponse:
    orchestrator = request.app.state.orchestrator
    try:
        state = orchestrator.run(body.query)
    except ProviderError as exc:
        # All configured LLM providers failed AND the agent-level document fallback
        # was bypassed (e.g. a new agent added without a try/except).
        logger.error("All LLM providers exhausted for query %r: %s", body.query, exc)
        raise HTTPException(
            status_code=503,
            detail="The LLM service is temporarily unavailable. Please try again later.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error processing query %r", body.query)
        raise HTTPException(status_code=500, detail="Internal server error.") from exc

    return QueryResponse(
        answer=state.final_answer,
        route=state.route,
        session_id=body.session_id,
    )


@router.get("/routes")
async def list_routes() -> dict[str, list[str]]:
    return {"routes": ["rag", "research", "memory"]}

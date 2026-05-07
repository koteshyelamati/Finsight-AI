from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=state.final_answer,
        route=state.route,
        session_id=body.session_id,
    )


@router.get("/routes")
async def list_routes() -> dict[str, list[str]]:
    return {"routes": ["rag", "research", "memory"]}

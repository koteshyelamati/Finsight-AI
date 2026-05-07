from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents import Orchestrator, AgentState


def _make_orchestrator_with_route(route: str) -> tuple[Orchestrator, MagicMock]:
    client = MagicMock()
    route_response = MagicMock()
    route_response.content = [MagicMock(text=route)]
    answer_response = MagicMock()
    answer_response.content = [MagicMock(text="Answer.", type="text")]
    client.messages.create.side_effect = [route_response, answer_response]
    return Orchestrator(client), client


@pytest.mark.parametrize("route", ["rag", "research", "memory"])
def test_routing_valid_routes(route):
    orchestrator, client = _make_orchestrator_with_route(route)
    state = AgentState(query="Tell me about AAPL earnings.")
    result = orchestrator.route(state)
    assert result == route
    assert state.route == route


def test_routing_fallback_on_invalid_response():
    orchestrator, client = _make_orchestrator_with_route("unknown_route")
    state = AgentState(query="Tell me about AAPL earnings.")
    result = orchestrator.route(state)
    assert result == "rag"


def test_orchestrator_run_calls_correct_agent():
    orchestrator, client = _make_orchestrator_with_route("rag")
    state = orchestrator.run("What is Apple's P/E ratio?")
    assert state.route == "rag"
    assert state.final_answer


def test_orchestrator_appends_chat_history():
    orchestrator, _ = _make_orchestrator_with_route("rag")
    state = orchestrator.run("What is Apple's P/E ratio?")
    roles = [m["role"] for m in state.chat_history]
    assert "user" in roles
    assert "assistant" in roles

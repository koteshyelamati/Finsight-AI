from __future__ import annotations

from unittest.mock import MagicMock


from agents import ResearchAgent, MemoryAgent, AgentState


def _make_text_response(text: str) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text=text, type="text")]
    client.messages.create.return_value = response
    return client


def test_research_agent_returns_non_empty_answer():
    client = _make_text_response("Revenue grew 12% YoY driven by cloud services.")
    agent = ResearchAgent(client)
    state = AgentState(query="Analyze Microsoft's cloud revenue growth.")
    result = agent.run(state)
    assert result.final_answer == "Revenue grew 12% YoY driven by cloud services."


def test_research_agent_passes_memory_context():
    client = _make_text_response("Analysis result.")
    agent = ResearchAgent(client)
    state = AgentState(query="Compare MSFT vs GOOG margins.", memory_context="User asked about MSFT last session.")
    agent.run(state)

    call_kwargs = client.messages.create.call_args.kwargs
    assert "User asked about MSFT last session." in call_kwargs["system"]


def test_memory_agent_returns_answer(tmp_path, monkeypatch):
    import agents.memory_agent as ma
    monkeypatch.setattr(ma, "MEMORY_DIR", tmp_path)

    client = _make_text_response("Based on past queries, AAPL has been trending positively.")
    agent = MemoryAgent(client)
    state = AgentState(query="How has Apple been doing lately?")
    result = agent.run(state)

    assert result.final_answer
    assert list(tmp_path.glob("*.json")), "Memory file should have been written"


def test_llm_answer_not_empty(mock_anthropic_client):
    from agents import RAGAgent
    agent = RAGAgent(mock_anthropic_client)
    state = AgentState(query="What is the debt-to-equity ratio of Tesla?")
    result = agent.run(state)
    assert isinstance(result.final_answer, str)
    assert len(result.final_answer) > 0

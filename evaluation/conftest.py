from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import anthropic
from agents import Orchestrator, AgentState


@pytest.fixture(scope="session")
def mock_anthropic_client():
    client = MagicMock(spec=anthropic.Anthropic)
    message = MagicMock()
    message.content = [MagicMock(text="Mock financial answer.", type="text")]
    client.messages.create.return_value = message
    return client


@pytest.fixture
def orchestrator(mock_anthropic_client):
    return Orchestrator(mock_anthropic_client)


@pytest.fixture
def sample_state():
    return AgentState(query="What was Apple's revenue in Q2 2024?")


@pytest.fixture
def sample_docs():
    return [
        {
            "content": "Apple reported Q2 2024 revenue of $90.8 billion, up 5% year-over-year.",
            "metadata": {"source": "earnings_call_q2.txt", "ticker": "AAPL"},
        },
        {
            "content": "iPhone revenue contributed $45.9 billion to total Q2 revenue.",
            "metadata": {"source": "earnings_call_q2.txt", "ticker": "AAPL"},
        },
    ]

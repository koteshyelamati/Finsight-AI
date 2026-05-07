from __future__ import annotations

import logging
from typing import Any, Literal

from .state import AgentState
from .rag_agent import RAGAgent
from .research_agent import ResearchAgent
from .memory_agent import MemoryAgent

logger = logging.getLogger(__name__)

ROUTING_PROMPT = """You are a financial AI router. Given the user query below, decide which agent should handle it.

Routes:
- "rag": answer from indexed financial documents (10-K, earnings calls, reports)
- "research": perform deeper analysis or synthesis across multiple topics
- "memory": recall or update information from past conversations

Query: {query}

Respond with exactly one word: rag, research, or memory."""


class Orchestrator:
    def __init__(self, client: Any, model: str = "claude-sonnet-4-6") -> None:
        self.client = client
        self.model = model
        self.rag_agent = RAGAgent(client, model)
        self.research_agent = ResearchAgent(client, model)
        self.memory_agent = MemoryAgent(client, model)

    def route(self, state: AgentState) -> Literal["rag", "research", "memory"]:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": ROUTING_PROMPT.format(query=state.query)}],
            )
            route = response.content[0].text.strip().lower()
            if route not in {"rag", "research", "memory"}:
                route = "rag"
        except Exception as exc:
            logger.error(
                "LLM routing call failed (%s: %s); defaulting to route='rag'.",
                type(exc).__name__, exc,
            )
            route = "rag"
        logger.info("Routed query to: %s", route)
        state.route = route
        return route  # type: ignore[return-value]

    def run(self, query: str) -> AgentState:
        state = AgentState(query=query)
        route = self.route(state)

        if route == "rag":
            state = self.rag_agent.run(state)
        elif route == "research":
            state = self.research_agent.run(state)
        else:
            state = self.memory_agent.run(state)

        state.add_message("user", query)
        state.add_message("assistant", state.final_answer)
        return state

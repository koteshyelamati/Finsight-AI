from __future__ import annotations

import logging
from typing import Any

from .state import AgentState

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """You are a senior financial research analyst. Perform a thorough analysis of the user's query. Break down complex financial topics, cite relevant metrics, and provide actionable insights. Use the memory context below if provided.

Memory Context:
{memory_context}"""


class ResearchAgent:
    def __init__(self, client: Any, model: str = "claude-sonnet-4-6") -> None:
        self.client = client
        self.model = model

    def _build_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "fetch_filing",
                "description": "Fetch a specific SEC filing or earnings call transcript by ticker and period.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol"},
                        "period": {"type": "string", "description": "Fiscal period, e.g. Q2-2024"},
                        "filing_type": {"type": "string", "enum": ["10-K", "10-Q", "earnings"]},
                    },
                    "required": ["ticker", "filing_type"],
                },
            }
        ]

    def run(self, state: AgentState) -> AgentState:
        messages: list[dict[str, Any]] = [{"role": "user", "content": state.query}]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=RESEARCH_SYSTEM_PROMPT.format(memory_context=state.memory_context or "None"),
                tools=self._build_tools(),
                messages=messages,
            )
        except Exception as exc:
            logger.error(
                "LLM call failed in ResearchAgent (%s: %s).",
                type(exc).__name__, exc,
            )
            state.final_answer = (
                "The research service is temporarily unavailable. "
                "Please try again later or check your LLM provider configuration."
            )
            return state

        result_parts: list[str] = []
        for block in response.content:
            if block.type == "text" and block.text:
                result_parts.append(block.text)
            elif block.type == "tool_use":
                logger.info("Tool call: %s(%s)", block.name, block.input)
                state.research_results.append({"tool": block.name, "input": block.input})

        state.final_answer = "\n".join(result_parts)
        return state

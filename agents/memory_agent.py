from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import anthropic

from .state import AgentState

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).parent.parent / "data" / "memory"

MEMORY_SYSTEM_PROMPT = """You are a memory-aware financial assistant. You have access to the user's past conversation summaries. Use them to provide contextually relevant answers.

Past summaries:
{summaries}"""


class MemoryAgent:
    def __init__(self, client: anthropic.Anthropic, model: str = "claude-sonnet-4-6") -> None:
        self.client = client
        self.model = model
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    def _load_summaries(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for fp in sorted(MEMORY_DIR.glob("*.json"))[-10:]:
            try:
                summaries.append(json.loads(fp.read_text()))
            except json.JSONDecodeError:
                logger.warning("Corrupt memory file: %s", fp)
        return summaries

    def _save_summary(self, state: AgentState) -> None:
        import time
        payload = {"timestamp": time.time(), "query": state.query, "answer": state.final_answer}
        dest = MEMORY_DIR / f"{int(payload['timestamp'])}.json"
        dest.write_text(json.dumps(payload, indent=2))

    def run(self, state: AgentState) -> AgentState:
        summaries = self._load_summaries()
        summary_text = "\n\n".join(
            f"[{s.get('timestamp', '')}] Q: {s.get('query', '')} → A: {s.get('answer', '')}"
            for s in summaries
        ) or "No previous summaries."

        state.memory_context = summary_text

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=MEMORY_SYSTEM_PROMPT.format(summaries=summary_text),
            messages=[{"role": "user", "content": state.query}],
        )
        state.final_answer = response.content[0].text
        self._save_summary(state)
        return state

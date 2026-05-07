from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    query: str
    chat_history: list[dict[str, str]] = field(default_factory=list)
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    research_results: list[dict[str, Any]] = field(default_factory=list)
    memory_context: str = ""
    final_answer: str = ""
    route: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        self.chat_history.append({"role": role, "content": content})

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "chat_history": self.chat_history,
            "retrieved_docs": self.retrieved_docs,
            "research_results": self.research_results,
            "memory_context": self.memory_context,
            "final_answer": self.final_answer,
            "route": self.route,
            "metadata": self.metadata,
        }

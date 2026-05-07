from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContentBlock:
    """Unified content block — mirrors the Anthropic TextBlock/ToolUseBlock interface
    so agents need no changes when switching providers."""
    type: str = "text"
    text: str | None = None
    name: str | None = None       # tool_use only
    input: dict[str, Any] | None = None  # tool_use only


@dataclass
class LLMResponse:
    content: list[ContentBlock] = field(default_factory=list)


class ProviderError(Exception):
    """Raised by any LLMProvider on recoverable API failures (auth, billing, quota)."""


class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def default_model(self) -> str: ...

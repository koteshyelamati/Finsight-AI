from __future__ import annotations

import logging
from typing import Any

from .base import ContentBlock, LLMProvider, LLMResponse, ProviderError

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL) -> None:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("anthropic is required for the Anthropic provider: pip install anthropic")
        self._sdk = _anthropic
        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return self._model

    def complete(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        try:
            raw = self._client.messages.create(**kwargs)
        except self._sdk.AuthenticationError as exc:
            raise ProviderError(f"Anthropic authentication failed: {exc}") from exc
        except self._sdk.PermissionDeniedError as exc:
            raise ProviderError(f"Anthropic permission denied (check billing/credits): {exc}") from exc
        except self._sdk.RateLimitError as exc:
            raise ProviderError(f"Anthropic rate limit exceeded: {exc}") from exc
        except self._sdk.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error {exc.status_code}: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"Anthropic unexpected error: {exc}") from exc

        blocks: list[ContentBlock] = []
        for block in raw.content:
            block_type = getattr(block, "type", "text")
            if block_type == "tool_use":
                blocks.append(ContentBlock(
                    type="tool_use",
                    name=block.name,
                    input=block.input,
                ))
            else:
                blocks.append(ContentBlock(type="text", text=getattr(block, "text", "")))

        return LLMResponse(content=blocks)

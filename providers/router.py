from __future__ import annotations

import logging
from typing import Any

from .base import LLMProvider, LLMResponse, ProviderError

logger = logging.getLogger(__name__)


class _MessagesAdapter:
    """Exposes .messages.create() so agents require no code changes."""

    def __init__(self, router: LLMRouter) -> None:
        self._router = router

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> LLMResponse:
        return self._router._dispatch(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=system,
            tools=tools,
        )


class LLMRouter:
    """Routes to the primary provider; falls back to secondary on ProviderError.

    Primary:  Gemini  (when GEMINI_API_KEY is set)
    Fallback: Anthropic (when ANTHROPIC_API_KEY is set, optional)
    """

    def __init__(self, primary: LLMProvider, fallback: LLMProvider | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.messages = _MessagesAdapter(self)

    def _dispatch(self, **kwargs: Any) -> LLMResponse:
        try:
            return self.primary.complete(**kwargs)
        except ProviderError as primary_exc:
            if self.fallback is None:
                logger.error(
                    "Provider %s failed and no fallback is configured: %s",
                    self.primary.name, primary_exc,
                )
                raise

            logger.warning(
                "Provider %s failed (%s); retrying with fallback %s.",
                self.primary.name, primary_exc, self.fallback.name,
            )

        try:
            return self.fallback.complete(**kwargs)
        except ProviderError as fallback_exc:
            logger.error("Fallback provider %s also failed: %s", self.fallback.name, fallback_exc)
            raise ProviderError(
                f"All providers exhausted. Last error ({self.fallback.name}): {fallback_exc}"
            ) from fallback_exc

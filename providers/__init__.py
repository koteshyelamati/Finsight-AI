from __future__ import annotations

import logging
import os

from .base import ContentBlock, LLMProvider, LLMResponse, ProviderError
from .router import LLMRouter

logger = logging.getLogger(__name__)


def build_llm_router() -> LLMRouter:
    """Build the LLM router from environment variables.

    Provider priority:
      1. Gemini  — primary when GEMINI_API_KEY is set (recommended)
      2. Anthropic — fallback when ANTHROPIC_API_KEY is set (optional)

    At least one key must be present; raises RuntimeError otherwise.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    primary: LLMProvider | None = None
    fallback: LLMProvider | None = None

    if gemini_key:
        from .gemini import GeminiProvider
        primary = GeminiProvider(api_key=gemini_key, model=gemini_model)
        logger.info("Primary LLM provider: Gemini (%s)", gemini_model)

    if anthropic_key:
        from .anthropic import AnthropicProvider
        provider = AnthropicProvider(api_key=anthropic_key, model=anthropic_model)
        if primary is None:
            primary = provider
            logger.info("Primary LLM provider: Anthropic (%s)", anthropic_model)
        else:
            fallback = provider
            logger.info("Fallback LLM provider: Anthropic (%s)", anthropic_model)

    if primary is None:
        raise RuntimeError(
            "No LLM provider configured. "
            "Set GEMINI_API_KEY (primary) and/or ANTHROPIC_API_KEY (fallback) in your environment."
        )

    return LLMRouter(primary=primary, fallback=fallback)


__all__ = [
    "build_llm_router",
    "LLMRouter",
    "LLMProvider",
    "LLMResponse",
    "ContentBlock",
    "ProviderError",
]

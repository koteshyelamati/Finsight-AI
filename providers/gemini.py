from __future__ import annotations

import logging
from typing import Any

from .base import ContentBlock, LLMProvider, LLMResponse, ProviderError

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


def _json_schema_to_gemini(schema: dict[str, Any]) -> Any:
    """Translate JSON Schema (Anthropic input_schema format) to google.genai Schema."""
    from google.genai import types

    type_map = {
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "object": "OBJECT",
        "array": "ARRAY",
    }
    raw_type = schema.get("type", "string")
    kwargs: dict[str, Any] = {"type": type_map.get(raw_type.lower(), "STRING")}

    if "description" in schema:
        kwargs["description"] = schema["description"]
    if "enum" in schema:
        kwargs["enum"] = schema["enum"]
    if raw_type == "object" and "properties" in schema:
        kwargs["properties"] = {k: _json_schema_to_gemini(v) for k, v in schema["properties"].items()}
        if "required" in schema:
            kwargs["required"] = schema["required"]
    if raw_type == "array" and "items" in schema:
        kwargs["items"] = _json_schema_to_gemini(schema["items"])

    return types.Schema(**kwargs)


def _translate_tools(tools: list[dict[str, Any]]) -> Any:
    from google.genai import types

    declarations = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters=_json_schema_to_gemini(t["input_schema"]),
        )
        for t in tools
    ]
    return types.Tool(function_declarations=declarations)


def _translate_messages(messages: list[dict[str, Any]]) -> list[Any]:
    from google.genai import types

    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        body = msg["content"]
        if isinstance(body, str):
            parts = [types.Part.from_text(text=body)]
        else:
            parts = [
                types.Part.from_text(text=item["text"])
                for item in body
                if isinstance(item, dict) and item.get("type") == "text"
            ]
        contents.append(types.Content(role=role, parts=parts))
    return contents


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL) -> None:
        try:
            from google import genai  # noqa: F401
        except ImportError:
            raise ImportError("google-genai is required for the Gemini provider: pip install google-genai")
        from google import genai as _genai
        self._client = _genai.Client(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return "gemini"

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
        from google.genai import types

        config_kwargs: dict[str, Any] = {"max_output_tokens": max_tokens}
        if system:
            config_kwargs["system_instruction"] = system
        if tools:
            config_kwargs["tools"] = [_translate_tools(tools)]

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=_translate_messages(messages),
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            raise ProviderError(f"Gemini API error: {exc}") from exc

        blocks: list[ContentBlock] = []
        for candidate in response.candidates or []:
            for part in candidate.content.parts or []:
                fc = getattr(part, "function_call", None)
                if fc:
                    blocks.append(ContentBlock(
                        type="tool_use",
                        name=fc.name,
                        input=dict(fc.args) if fc.args else {},
                    ))
                elif getattr(part, "text", None):
                    blocks.append(ContentBlock(type="text", text=part.text))

        if not blocks:
            fallback_text = getattr(response, "text", "") or ""
            blocks = [ContentBlock(type="text", text=fallback_text)]

        return LLMResponse(content=blocks)

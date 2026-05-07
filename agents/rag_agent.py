from __future__ import annotations

import logging
from typing import Any

from .state import AgentState

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a financial analyst assistant. Use the retrieved documents below to answer the user's question accurately and concisely. If the documents do not contain enough information, say so.

Retrieved Documents:
{context}"""


def _doc_fallback_answer(docs: list[dict[str, Any]], query: str) -> str:
    """Return retrieved document excerpts verbatim when LLM synthesis is unavailable.

    Deterministic: same docs + same query always produce the same output.
    """
    if not docs:
        return (
            "The LLM service is temporarily unavailable and no indexed documents "
            "were found for your query. Please try again later, or run `make ingest` "
            "to load financial documents into the vector store."
        )
    header = (
        "The LLM synthesis service is temporarily unavailable. "
        "Below are the most relevant document passages retrieved for your query:\n"
    )
    excerpts = []
    for doc in docs[:3]:
        source = doc.get("metadata", {}).get("source", "Document")
        content = doc["content"][:600].rstrip()
        excerpts.append(f"[{source}]\n{content}")
    return header + "\n\n".join(excerpts)


class RAGAgent:
    def __init__(self, client: Any, model: str = "claude-sonnet-4-6") -> None:
        self.client = client
        self.model = model
        self._vector_store: Any = None

    def set_vector_store(self, store: Any) -> None:
        self._vector_store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._vector_store is None:
            logger.warning("No vector store attached; returning empty context.")
            return []
        results = self._vector_store.similarity_search(query, k=top_k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]

    def run(self, state: AgentState) -> AgentState:
        docs = self.retrieve(state.query)
        state.retrieved_docs = docs

        context = "\n\n---\n\n".join(d["content"] for d in docs) if docs else "No documents retrieved."

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=RAG_SYSTEM_PROMPT.format(context=context),
                messages=[{"role": "user", "content": state.query}],
            )
            state.final_answer = response.content[0].text
        except Exception as exc:
            logger.error(
                "LLM call failed in RAGAgent (%s: %s); returning document fallback.",
                type(exc).__name__, exc,
            )
            state.final_answer = _doc_fallback_answer(docs, state.query)
        return state

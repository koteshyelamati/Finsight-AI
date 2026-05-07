from __future__ import annotations

import logging
from typing import Any

import anthropic

from .state import AgentState

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a financial analyst assistant. Use the retrieved documents below to answer the user's question accurately and concisely. If the documents do not contain enough information, say so.

Retrieved Documents:
{context}"""


class RAGAgent:
    def __init__(self, client: anthropic.Anthropic, model: str = "claude-sonnet-4-6") -> None:
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

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=RAG_SYSTEM_PROMPT.format(context=context),
            messages=[{"role": "user", "content": state.query}],
        )
        state.final_answer = response.content[0].text
        return state

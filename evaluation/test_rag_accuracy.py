from __future__ import annotations

from unittest.mock import MagicMock


from agents import RAGAgent


def test_rag_returns_answer(mock_anthropic_client, sample_state, sample_docs):
    agent = RAGAgent(mock_anthropic_client)
    store = MagicMock()
    store.similarity_search.return_value = [
        MagicMock(page_content=d["content"], metadata=d["metadata"]) for d in sample_docs
    ]
    agent.set_vector_store(store)

    result = agent.run(sample_state)

    assert result.final_answer
    assert len(result.retrieved_docs) == len(sample_docs)


def test_rag_no_vector_store_returns_answer(mock_anthropic_client, sample_state):
    agent = RAGAgent(mock_anthropic_client)
    result = agent.run(sample_state)

    assert result.final_answer
    assert result.retrieved_docs == []


def test_rag_context_passed_to_llm(mock_anthropic_client, sample_state, sample_docs):
    agent = RAGAgent(mock_anthropic_client)
    store = MagicMock()
    store.similarity_search.return_value = [
        MagicMock(page_content=d["content"], metadata=d["metadata"]) for d in sample_docs
    ]
    agent.set_vector_store(store)
    agent.run(sample_state)

    call_kwargs = mock_anthropic_client.messages.create.call_args
    assert sample_docs[0]["content"] in call_kwargs.kwargs["system"]


def test_rag_retrieves_correct_top_k(mock_anthropic_client, sample_state):
    agent = RAGAgent(mock_anthropic_client)
    store = MagicMock()
    store.similarity_search.return_value = []
    agent.set_vector_store(store)
    agent.run(sample_state)

    store.similarity_search.assert_called_once_with(sample_state.query, k=5)

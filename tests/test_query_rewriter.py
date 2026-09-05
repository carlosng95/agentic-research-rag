import pytest

from agentic_research_rag.memory import ConversationMemory
from agentic_research_rag.query_rewriter import QueryRewriter
from tests.fakes import FakeLLMProvider


def test_query_rewriter_returns_original_query_without_history():
    llm = FakeLLMProvider(
        response = "This should never be used."
    )

    rewriter = QueryRewriter(
        llm = llm,
    )

    memory = ConversationMemory(
        max_turns = 5,
    )

    query = "How does semantic search retrieve documents?"

    result = rewriter.rewrite(
        query = query,
        memory = memory,
    )

    assert result == query
    assert llm.prompts == []


def test_query_rewriter_uses_history():
    llm = FakeLLMProvider(
        response = "How does semantic search differ from lexical document retrieval?"
    )

    rewriter = QueryRewriter(
        llm = llm,
    )

    memory = ConversationMemory(
        max_turns = 5,
    )

    memory.add_user(
        "How does semantic search retrieve documents?"
    )

    memory.add_assistant(
        "Semantic search retrieves documents using vector similarity."
    )

    result = rewriter.rewrite(
        query = "And how does it differ from lexical search?",
        memory = memory,
    )

    assert result == (
        "How does semantic search differ from lexical document retrieval?"
    )

    assert len(llm.prompts) == 1


def test_query_rewriter_prompt_contains_history_and_current_query():
    llm = FakeLLMProvider(
        response = "Standalone query"
    )

    rewriter = QueryRewriter(
        llm = llm,
    )

    memory = ConversationMemory(
        max_turns = 5,
    )

    memory.add_user(
        "How does semantic search retrieve documents?"
    )

    memory.add_assistant(
        "Semantic search retrieves documents using vector similarity."
    )

    query = "And how does it differ from lexical search?"

    rewriter.rewrite(
        query = query,
        memory = memory,
    )

    prompt = llm.prompts[0]

    assert "How does semantic search retrieve documents?" in prompt
    assert "Semantic search retrieves documents using vector similarity." in prompt
    assert query in prompt


def test_query_rewriter_falls_back_to_original_query_when_llm_returns_empty():
    llm = FakeLLMProvider(
        response = "   "
    )

    rewriter = QueryRewriter(
        llm = llm,
    )

    memory = ConversationMemory(
        max_turns = 5,
    )

    memory.add_user(
        "How does semantic search retrieve documents?"
    )

    query = "And how does it differ from lexical search?"

    result = rewriter.rewrite(
        query = query,
        memory = memory,
    )

    assert result == query


def test_query_rewriter_rejects_empty_query():
    llm = FakeLLMProvider(
        response = "Anything"
    )

    rewriter = QueryRewriter(
        llm = llm,
    )

    memory = ConversationMemory(
        max_turns = 5,
    )

    with pytest.raises(
        ValueError,
        match = "Query cannot be empty",
    ):
        rewriter.rewrite(
            query = "   ",
            memory = memory,
        )

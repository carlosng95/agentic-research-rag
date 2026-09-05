import pytest

from agentic_research_rag.evaluation.sufficiency import SufficiencyEvaluator
from agentic_research_rag.types import ResearchResponse, Source
from tests.fakes import FakeLLMProvider


def make_response_with_sources() -> ResearchResponse:
    return ResearchResponse(
        answer = "Paper-based answer.",
        sources = [
            Source(
                type = "paper",
                ref = "paper_a.pdf",
                locator = "page 2",
                snippet = "Semantic retrieval represents documents using dense vectors.",
            ),
            Source(
                type = "paper",
                ref = "paper_b.pdf",
                locator = "page 5",
                snippet = "BM25 ranks documents using lexical term statistics.",
            ),
        ],
    )


def test_sufficiency_returns_true_for_sufficient_evidence():
    llm = FakeLLMProvider(
        response = "SUFFICIENT"
    )

    evaluator = SufficiencyEvaluator(
        llm = llm,
    )

    response = make_response_with_sources()

    result = evaluator.is_sufficient(
        query = "How does hybrid retrieval combine semantic and lexical search?",
        response = response,
    )

    assert result is True
    assert len(llm.prompts) == 1


def test_sufficiency_returns_false_for_insufficient_evidence():
    llm = FakeLLMProvider(
        response = "INSUFFICIENT"
    )

    evaluator = SufficiencyEvaluator(
        llm = llm,
    )

    response = make_response_with_sources()

    result = evaluator.is_sufficient(
        query = "What was the weather in Buenos Aires yesterday?",
        response = response,
    )

    assert result is False
    assert len(llm.prompts) == 1


def test_sufficiency_returns_false_without_sources():
    llm = FakeLLMProvider(
        response = "SUFFICIENT"
    )

    evaluator = SufficiencyEvaluator(
        llm = llm,
    )

    response = ResearchResponse(
        answer = "No evidence.",
        sources = [],
    )

    result = evaluator.is_sufficient(
        query = "What is semantic search?",
        response = response,
    )

    assert result is False
    assert llm.prompts == []


def test_sufficiency_prompt_contains_query_and_evidence():
    llm = FakeLLMProvider(
        response = "SUFFICIENT"
    )

    evaluator = SufficiencyEvaluator(
        llm = llm,
    )

    response = make_response_with_sources()

    query = "How does hybrid retrieval combine semantic and lexical search?"

    evaluator.is_sufficient(
        query = query,
        response = response,
    )

    prompt = llm.prompts[0]

    assert query in prompt
    assert "Semantic retrieval represents documents using dense vectors." in prompt
    assert "BM25 ranks documents using lexical term statistics." in prompt
    assert "paper_a.pdf" in prompt
    assert "paper_b.pdf" in prompt


def test_sufficiency_normalizes_llm_response():
    llm = FakeLLMProvider(
        response = "  sufficient.  "
    )

    evaluator = SufficiencyEvaluator(
        llm = llm,
    )

    response = make_response_with_sources()

    result = evaluator.is_sufficient(
        query = "What is semantic search?",
        response = response,
    )

    assert result is True


def test_sufficiency_rejects_unexpected_llm_decision():
    llm = FakeLLMProvider(
        response = "MAYBE"
    )

    evaluator = SufficiencyEvaluator(
        llm = llm,
    )

    response = make_response_with_sources()

    with pytest.raises(
        ValueError,
        match = "Unexpected sufficiency decision",
    ):
        evaluator.is_sufficient(
            query = "What is semantic search?",
            response = response,
        )


def test_sufficiency_rejects_empty_query():
    llm = FakeLLMProvider(
        response = "SUFFICIENT"
    )

    evaluator = SufficiencyEvaluator(
        llm = llm,
    )

    response = make_response_with_sources()

    with pytest.raises(
        ValueError,
        match = "Query cannot be empty",
    ):
        evaluator.is_sufficient(
            query = "   ",
            response = response,
        )

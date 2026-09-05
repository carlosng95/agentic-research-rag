from agentic_research_rag.synthesis.evidence import EvidenceSynthesizer
from agentic_research_rag.types import ResearchResponse, Source
from tests.fakes import FakeLLMProvider
import pytest

def make_paper_response() -> ResearchResponse:
    return ResearchResponse(
        answer = "Paper answer [SOURCE 1].",
        sources = [
            Source(
                type = "paper",
                ref = "paper_a.pdf",
                locator = "page 2",
                snippet = "Paper evidence A.",
            ),
            Source(
                type = "paper",
                ref = "paper_b.pdf",
                locator = "page 5",
                snippet = "Paper evidence B.",
            ),
        ],
    )


def make_web_response() -> ResearchResponse:
    return ResearchResponse(
        answer = "Web answer.",
        sources = [
            Source(
                type = "web",
                ref = "https://example.com/article",
                locator = "Example article",
            ),
            Source(
                type = "web",
                ref = "https://example.com/research",
                locator = "Research article",
            ),
        ],
    )


def test_synthesizer_combines_paper_and_web_sources():
    llm = FakeLLMProvider(
        response = "Combined answer [SOURCE 1][SOURCE 3]."
    )

    synthesizer = EvidenceSynthesizer(
        llm = llm,
    )

    response = synthesizer.synthesize(
        query = "What does the evidence show?",
        paper_response = make_paper_response(),
        web_response = make_web_response(),
    )

    assert len(response.sources) == 4

    assert response.sources[0].type == "paper"
    assert response.sources[1].type == "paper"
    assert response.sources[2].type == "web"
    assert response.sources[3].type == "web"


def test_synthesizer_preserves_source_order():
    llm = FakeLLMProvider(
        response = "Combined answer [SOURCE 1][SOURCE 3]."
    )

    synthesizer = EvidenceSynthesizer(
        llm = llm,
    )

    response = synthesizer.synthesize(
        query = "What does the evidence show?",
        paper_response = make_paper_response(),
        web_response = make_web_response(),
    )

    assert response.sources[0].ref == "paper_a.pdf"
    assert response.sources[1].ref == "paper_b.pdf"
    assert response.sources[2].ref == "https://example.com/article"
    assert response.sources[3].ref == "https://example.com/research"


def test_synthesizer_deduplicates_sources():
    duplicate_source = Source(
        type = "web",
        ref = "https://example.com/article",
        locator = "Example article",
    )

    paper_response = ResearchResponse(
        answer = "Paper answer.",
        sources = [
            duplicate_source,
        ],
    )

    web_response = ResearchResponse(
        answer = "Web answer.",
        sources = [
            duplicate_source,
        ],
    )

    llm = FakeLLMProvider(
        response = "Combined answer [SOURCE 1]."
    )

    synthesizer = EvidenceSynthesizer(
        llm = llm,
    )

    response = synthesizer.synthesize(
        query = "What does the evidence show?",
        paper_response = paper_response,
        web_response = web_response,
    )

    assert len(response.sources) == 1
    assert response.sources[0].ref == "https://example.com/article"


def test_synthesizer_prompt_contains_query_and_intermediate_answers():
    llm = FakeLLMProvider(
        response = "Final answer [SOURCE 1]."
    )

    synthesizer = EvidenceSynthesizer(
        llm = llm,
    )

    paper_response = make_paper_response()
    web_response = make_web_response()

    query = "What does the evidence show?"

    synthesizer.synthesize(
        query = query,
        paper_response = paper_response,
        web_response = web_response,
    )

    assert len(llm.prompts) == 1

    prompt = llm.prompts[0]

    assert query in prompt
    assert paper_response.answer in prompt
    assert web_response.answer in prompt


def test_synthesizer_prompt_contains_unified_sources():
    llm = FakeLLMProvider(
        response = "Final answer [SOURCE 1][SOURCE 3]."
    )

    synthesizer = EvidenceSynthesizer(
        llm = llm,
    )

    synthesizer.synthesize(
        query = "What does the evidence show?",
        paper_response = make_paper_response(),
        web_response = make_web_response(),
    )

    prompt = llm.prompts[0]

    assert "[SOURCE 1]" in prompt
    assert "[SOURCE 2]" in prompt
    assert "[SOURCE 3]" in prompt
    assert "[SOURCE 4]" in prompt

    assert "paper_a.pdf" in prompt
    assert "paper_b.pdf" in prompt
    assert "https://example.com/article" in prompt
    assert "https://example.com/research" in prompt


def test_synthesizer_includes_paper_snippets_in_prompt():
    llm = FakeLLMProvider(
        response = "Final answer [SOURCE 1]."
    )

    synthesizer = EvidenceSynthesizer(
        llm = llm,
    )

    synthesizer.synthesize(
        query = "What does the evidence show?",
        paper_response = make_paper_response(),
        web_response = make_web_response(),
    )

    prompt = llm.prompts[0]

    assert "Paper evidence A." in prompt
    assert "Paper evidence B." in prompt


def test_synthesizer_returns_llm_answer():
    generated_answer = (
        "Scientific papers provide historical evidence, "
        "while web sources provide recent information [SOURCE 1][SOURCE 3]."
    )

    llm = FakeLLMProvider(
        response = generated_answer,
    )

    synthesizer = EvidenceSynthesizer(
        llm = llm,
    )

    response = synthesizer.synthesize(
        query = "What does the evidence show?",
        paper_response = make_paper_response(),
        web_response = make_web_response(),
    )

    assert response.answer == generated_answer


def test_synthesizer_accepts_valid_citations():
    llm = FakeLLMProvider(
        response = "Combined evidence [SOURCE 1][SOURCE 4]."
    )

    synthesizer = EvidenceSynthesizer(
        llm = llm,
    )

    response = synthesizer.synthesize(
        query = "What does the evidence show?",
        paper_response = make_paper_response(),
        web_response = make_web_response(),
    )

    assert response.flags == []


def test_synthesizer_detects_invalid_citation():
    llm = FakeLLMProvider(
        response = "Combined evidence [SOURCE 9]."
    )

    synthesizer = EvidenceSynthesizer(
        llm = llm,
    )

    response = synthesizer.synthesize(
        query = "What does the evidence show?",
        paper_response = make_paper_response(),
        web_response = make_web_response(),
    )

    assert response.flags == [
        "invalid_source_9",
    ]


def test_synthesizer_detects_missing_citations():
    llm = FakeLLMProvider(
        response = "Combined answer without citations."
    )

    synthesizer = EvidenceSynthesizer(
        llm = llm,
    )

    response = synthesizer.synthesize(
        query = "What does the evidence show?",
        paper_response = make_paper_response(),
        web_response = make_web_response(),
    )

    assert response.flags == [
        "missing_citations",
    ]
    
def test_synthesizer_rejects_empty_query():
    llm = FakeLLMProvider(
        response = "Anything"
    )

    synthesizer = EvidenceSynthesizer(
        llm = llm,
    )

    with pytest.raises(
        ValueError,
        match = "Query cannot be empty",
    ):
        synthesizer.synthesize(
            query = "   ",
            paper_response = make_paper_response(),
            web_response = make_web_response(),
        )

    assert llm.prompts == []
    

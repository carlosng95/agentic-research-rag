from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from agentic_research_rag.assistant import (
    ResearchAssistant,
    ResearchResponse,
    Source,
    state_to_response,
)


def test_state_to_response_converts_paper_document() -> None:
    document = Document(
        page_content = (
            "  Local\n\nresearch   evidence.  "
        ),
        metadata = {
            "source": "research.pdf",
            "page_number": 4,
            "chunk_id": 10,
        },
    )

    response = state_to_response(
        state = {
            "final_answer": (
                "Grounded answer [SOURCE 1]."
            ),
            "final_documents": [
                document
            ],
            "cited_source_numbers": [
                1
            ],
            "tools_used": [
                "paper_retrieval"
            ],
            "flags": [],
            "sufficient": True,
            "citation_valid": True,
        }
    )

    assert isinstance(
        response,
        ResearchResponse,
    )

    assert response.answer == (
        "Grounded answer [SOURCE 1]."
    )

    assert response.sources == [
        Source(
            source_number = 1,
            type = "paper",
            ref = "research.pdf",
            locator = "page 4",
            snippet = (
                "Local research evidence."
            ),
        )
    ]

    assert (
        response.paper_evidence_sufficient
        is True
    )

    assert response.citation_valid is True


def test_state_to_response_converts_web_document() -> None:
    document = Document(
        page_content = "Web evidence.",
        metadata = {
            "source": (
                "https://example.com/article"
            ),
            "url": (
                "https://example.com/article"
            ),
            "title": "Article",
        },
    )

    response = state_to_response(
        state = {
            "final_answer": (
                "Web-supported answer "
                "[SOURCE 1]."
            ),
            "final_documents": [
                document
            ],
            "cited_source_numbers": [
                1
            ],
            "tools_used": [
                "paper_retrieval",
                "web_search",
            ],
            "flags": [
                "web_fallback_used"
            ],
            "sufficient": False,
            "citation_valid": True,
        }
    )

    source = response.sources[0]

    assert source.type == "web"

    assert source.ref == (
        "https://example.com/article"
    )

    assert source.locator == (
        "https://example.com/article"
    )

    assert (
        response.paper_evidence_sufficient
        is False
    )


def test_research_assistant_invokes_graph_with_thread_id() -> None:
    calls: dict[str, Any] = {}

    def run_graph(
        state: dict,
        config: dict,
    ) -> dict:
        calls["state"] = state
        calls["config"] = config

        return {
            "final_answer": (
                "Research answer."
            ),
            "final_documents": [],
            "cited_source_numbers": [],
            "tools_used": [],
            "flags": [],
            "sufficient": False,
            "citation_valid": True,
        }

    graph = RunnableLambda(
        run_graph
    )

    assistant = ResearchAssistant(
        graph = graph,
        thread_id = "conversation-123",
    )

    response = assistant.ask(
        "  Research question  "
    )

    assert response.answer == (
        "Research answer."
    )

    assert calls["state"] == {
        "question": "Research question",
        "messages": [],
    }

    assert calls['config']['configurable']['thread_id'] == 'conversation-123'


def test_research_assistant_rejects_empty_question() -> None:
    graph = RunnableLambda(
        lambda state: state
    )

    assistant = ResearchAssistant(
        graph = graph
    )

    with pytest.raises(
        ValueError
    ):
        assistant.ask(
            "   "
        )


def test_research_assistant_rejects_empty_thread_id() -> None:
    graph = RunnableLambda(
        lambda state: state
    )

    with pytest.raises(
        ValueError
    ):
        ResearchAssistant(
            graph = graph,
            thread_id = "   ",
        )
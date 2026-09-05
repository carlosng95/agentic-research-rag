from typing import Any

import pytest
from langchain_core.tools import StructuredTool

from agentic_research_rag.tools.web_search import (
    search_web,
    web_results_to_documents,
)


def make_fake_tool(
    response: dict[str, Any],
) -> StructuredTool:
    def fake_search(
        query: str,
    ) -> dict[str, Any]:
        return response

    return StructuredTool.from_function(
        func = fake_search,
        name = "fake_web_search",
        description = "Fake web search for tests.",
    )


def test_web_results_to_documents() -> None:
    response = {
        "results": [
            {
                "title": "First article",
                "url": "https://example.com/first",
                "content": "First web evidence.",
                "score": 0.9,
            },
            {
                "title": "Second article",
                "url": "https://example.com/second",
                "content": "Second web evidence.",
                "score": 0.8,
            },
        ]
    }

    documents = web_results_to_documents(
        response = response
    )

    assert len(documents) == 2

    assert documents[0].page_content == (
        "First web evidence."
    )

    assert documents[0].metadata == {
        "source": "https://example.com/first",
        "url": "https://example.com/first",
        "title": "First article",
        "web_rank": 1,
        "search_score": 0.9,
    }

    assert documents[1].metadata["web_rank"] == 2


def test_web_results_to_documents_skips_invalid_results() -> None:
    response = {
        "results": [
            {
                "title": "Missing URL",
                "url": "",
                "content": "Evidence.",
            },
            {
                "title": "Missing content",
                "url": "https://example.com/empty",
                "content": "",
            },
            {
                "title": "Valid",
                "url": "https://example.com/valid",
                "content": "Valid evidence.",
            },
        ]
    }

    documents = web_results_to_documents(
        response = response
    )

    assert len(documents) == 1

    assert (
        documents[0].metadata["url"]
        == "https://example.com/valid"
    )


def test_web_results_to_documents_deduplicates_urls() -> None:
    response = {
        "results": [
            {
                "title": "First",
                "url": "https://example.com/article",
                "content": "First evidence.",
            },
            {
                "title": "Duplicate",
                "url": "https://example.com/article",
                "content": "Duplicate evidence.",
            },
        ]
    }

    documents = web_results_to_documents(
        response = response
    )

    assert len(documents) == 1
    assert documents[0].page_content == "First evidence."


def test_search_web_invokes_langchain_tool() -> None:
    tool = make_fake_tool(
        response = {
            "results": [
                {
                    "title": "Article",
                    "url": "https://example.com/article",
                    "content": "Web evidence.",
                    "score": 0.7,
                }
            ]
        }
    )

    documents = search_web(
        query = "hybrid retrieval",
        tool = tool,
    )

    assert len(documents) == 1

    assert (
        documents[0].metadata["title"]
        == "Article"
    )


def test_search_web_rejects_empty_query() -> None:
    tool = make_fake_tool(
        response = {
            "results": [],
        }
    )

    with pytest.raises(ValueError):
        search_web(
            query = "   ",
            tool = tool,
        )


def test_web_results_requires_results_list() -> None:
    with pytest.raises(TypeError):
        web_results_to_documents(
            response = {
                "results": "invalid",
            }
        )
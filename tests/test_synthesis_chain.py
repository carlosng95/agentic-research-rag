from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
)

from agentic_research_rag.chains.synthesis import (
    build_synthesis_chain,
    format_synthesis_context,
)


class CountingFakeListChatModel(FakeListChatModel):
    calls: int = 0

    def _call(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        self.calls += 1

        return super()._call(
            *args,
            **kwargs,
        )


def make_paper_document(
    chunk_id: int,
    source: str,
    page_number: int,
    text: str,
) -> Document:
    return Document(
        page_content = text,
        metadata = {
            "chunk_id": chunk_id,
            "source": source,
            "page_number": page_number,
        },
    )


def make_web_document(
    title: str,
    url: str,
    text: str,
) -> Document:
    return Document(
        page_content = text,
        metadata = {
            "source": url,
            "url": url,
            "title": title,
        },
    )


def test_format_synthesis_context_uses_unified_source_numbers() -> None:
    paper_documents = [
        make_paper_document(
            chunk_id = 10,
            source = "local.pdf",
            page_number = 2,
            text = "Local evidence.",
        )
    ]

    web_documents = [
        make_web_document(
            title = "Web article",
            url = "https://example.com/article",
            text = "Web evidence.",
        )
    ]

    context = format_synthesis_context(
        paper_documents = paper_documents,
        web_documents = web_documents,
    )

    assert "[SOURCE 1]" in context
    assert "Type: local document" in context
    assert "Document: local.pdf" in context
    assert "Page: 2" in context
    assert "Chunk ID: 10" in context

    assert "[SOURCE 2]" in context
    assert "Type: web" in context
    assert "Title: Web article" in context
    assert "URL: https://example.com/article" in context


def test_synthesis_chain_returns_answer_and_combined_documents() -> None:
    paper_document = make_paper_document(
        chunk_id = 10,
        source = "local.pdf",
        page_number = 2,
        text = "Local evidence.",
    )

    web_document = make_web_document(
        title = "Web article",
        url = "https://example.com/article",
        text = "Complementary web evidence.",
    )

    model = CountingFakeListChatModel(
        responses = [
            (
                "The combined evidence supports the "
                "answer [SOURCE 1] [SOURCE 2]."
            )
        ]
    )

    chain = build_synthesis_chain(
        model = model
    )

    result = chain.invoke(
        {
            "question": "Research question",
            "paper_documents": [
                paper_document
            ],
            "web_documents": [
                web_document
            ],
        }
    )

    assert result["answer"] == (
        "The combined evidence supports the "
        "answer [SOURCE 1] [SOURCE 2]."
    )

    assert result["documents"] == [
        paper_document,
        web_document,
    ]

    assert model.calls == 1


def test_synthesis_chain_supports_web_only_evidence() -> None:
    web_document = make_web_document(
        title = "Web article",
        url = "https://example.com/article",
        text = "Web evidence.",
    )

    model = CountingFakeListChatModel(
        responses = [
            "Answer based on web evidence [SOURCE 1]."
        ]
    )

    chain = build_synthesis_chain(
        model = model
    )

    result = chain.invoke(
        {
            "question": "Research question",
            "paper_documents": [],
            "web_documents": [
                web_document
            ],
        }
    )

    assert result["documents"] == [
        web_document
    ]

    assert model.calls == 1


def test_synthesis_chain_skips_model_without_evidence() -> None:
    model = CountingFakeListChatModel(
        responses = [
            "This response should never be used."
        ]
    )

    chain = build_synthesis_chain(
        model = model
    )

    result = chain.invoke(
        {
            "question": "Unknown question",
            "paper_documents": [],
            "web_documents": [],
        }
    )

    assert result["answer"] == (
        "No sufficient evidence was found in the "
        "available documents or web sources."
    )

    assert result["documents"] == []
    assert model.calls == 0
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
)
from langchain_core.retrievers import BaseRetriever

from agentic_research_rag.chains.rag import (
    build_rag_chain,
    format_documents,
)


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(
        self,
        query: str,
    ) -> list[Document]:
        return self.documents


def make_document(
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


def test_format_documents() -> None:
    documents = [
        make_document(
            chunk_id = 10,
            source = "document_a.pdf",
            page_number = 2,
            text = "First evidence.",
        ),
        make_document(
            chunk_id = 20,
            source = "document_b.pdf",
            page_number = 5,
            text = "Second evidence.",
        ),
    ]

    context = format_documents(
        documents = documents
    )

    assert "[SOURCE 1]" in context
    assert "Document: document_a.pdf" in context
    assert "Page: 2" in context
    assert "Chunk ID: 10" in context
    assert "First evidence." in context

    assert "[SOURCE 2]" in context
    assert "Document: document_b.pdf" in context
    assert "Page: 5" in context
    assert "Chunk ID: 20" in context
    assert "Second evidence." in context


def test_rag_chain_returns_answer_and_documents() -> None:
    documents = [
        make_document(
            chunk_id = 0,
            source = "retrieval.pdf",
            page_number = 1,
            text = (
                "Hybrid retrieval combines semantic "
                "and lexical search."
            ),
        )
    ]

    retriever = StaticRetriever(
        documents = documents
    )

    model = FakeListChatModel(
        responses = [
            (
                "Hybrid retrieval combines both "
                "approaches [SOURCE 1]."
            )
        ]
    )

    chain = build_rag_chain(
        retriever = retriever,
        model = model,
    )

    result = chain.invoke(
        "What is hybrid retrieval?"
    )

    assert result["answer"] == (
        "Hybrid retrieval combines both "
        "approaches [SOURCE 1]."
    )

    assert result["documents"] == documents


def test_rag_chain_skips_model_when_no_documents() -> None:
    retriever = StaticRetriever(
        documents = []
    )

    model = FakeListChatModel(
        responses = [
            "This response should never be used."
        ]
    )

    chain = build_rag_chain(
        retriever = retriever,
        model = model,
    )

    result = chain.invoke(
        "Unknown question"
    )

    assert result["answer"] == (
        "No relevant information was found "
        "in the available documents."
    )

    assert result["documents"] == []

    assert model.i == 0


def test_rag_chain_assigns_source_numbers_by_rank() -> None:
    documents = [
        make_document(
            chunk_id = 90,
            source = "first.pdf",
            page_number = 8,
            text = "First ranked evidence.",
        ),
        make_document(
            chunk_id = 12,
            source = "second.pdf",
            page_number = 3,
            text = "Second ranked evidence.",
        ),
    ]

    retriever = StaticRetriever(
        documents = documents
    )

    model = FakeListChatModel(
        responses = [
            "Answer [SOURCE 1] [SOURCE 2]."
        ]
    )

    chain = build_rag_chain(
        retriever = retriever,
        model = model,
    )

    result = chain.invoke(
        "query"
    )

    assert (
        result["documents"][0]
        .metadata["chunk_id"]
        == 90
    )

    assert (
        result["documents"][1]
        .metadata["chunk_id"]
        == 12
    )
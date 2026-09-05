from agentic_research_rag.context_builder import build_context
from agentic_research_rag.types import Chunk


def make_chunk(
    chunk_id: int,
    document_name: str,
    page_number: int,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id = chunk_id,
        document_name = document_name,
        page_number = page_number,
        text = text,
        score = 0.95,
    )


def test_build_context_returns_empty_for_no_chunks():
    context = build_context(
        chunks = []
    )

    assert context == ""


def test_build_context_contains_source_number():
    chunks = [
        make_chunk(
            chunk_id = 10,
            document_name = "paper.pdf",
            page_number = 3,
            text = "Document evidence.",
        ),
    ]

    context = build_context(
        chunks = chunks
    )

    assert "[SOURCE 1]" in context


def test_build_context_contains_chunk_metadata():
    chunks = [
        make_chunk(
            chunk_id = 42,
            document_name = "retrieval.pdf",
            page_number = 7,
            text = "Hybrid retrieval combines semantic and lexical search.",
        ),
    ]

    context = build_context(
        chunks = chunks
    )

    assert "retrieval.pdf" in context
    assert "7" in context
    assert "42" in context
    assert "Hybrid retrieval combines semantic and lexical search." in context


def test_build_context_numbers_sources_sequentially():
    chunks = [
        make_chunk(
            chunk_id = 10,
            document_name = "paper_a.pdf",
            page_number = 1,
            text = "Evidence A",
        ),
        make_chunk(
            chunk_id = 50,
            document_name = "paper_b.pdf",
            page_number = 2,
            text = "Evidence B",
        ),
        make_chunk(
            chunk_id = 99,
            document_name = "paper_c.pdf",
            page_number = 3,
            text = "Evidence C",
        ),
    ]

    context = build_context(
        chunks = chunks
    )

    assert "[SOURCE 1]" in context
    assert "[SOURCE 2]" in context
    assert "[SOURCE 3]" in context


def test_build_context_uses_source_position_not_chunk_id():
    chunks = [
        make_chunk(
            chunk_id = 500,
            document_name = "paper.pdf",
            page_number = 1,
            text = "Evidence",
        ),
    ]

    context = build_context(
        chunks = chunks
    )

    assert "[SOURCE 1]" in context
    assert "[SOURCE 500]" not in context


def test_build_context_does_not_include_retrieval_score():
    chunks = [
        make_chunk(
            chunk_id = 1,
            document_name = "paper.pdf",
            page_number = 1,
            text = "Evidence",
        ),
    ]

    context = build_context(
        chunks = chunks
    )

    assert "0.95" not in context
    assert "Score:" not in context
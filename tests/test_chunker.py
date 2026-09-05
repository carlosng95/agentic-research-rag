import pytest

from agentic_research_rag.ingestion.chunker import chunk_page, chunk_pages, chunk_text
from agentic_research_rag.types import Page


def test_chunk_text_returns_single_chunk_when_text_fits():
    text = "abcdefghij"

    chunks = chunk_text(
        text = text,
        chunk_size = 20,
        overlap = 5,
    )

    assert chunks == [
        "abcdefghij",
    ]


def test_chunk_text_splits_text_using_chunk_size():
    text = "abcdefghij"

    chunks = chunk_text(
        text = text,
        chunk_size = 4,
        overlap = 0,
    )

    assert chunks == [
        "abcd",
        "efgh",
        "ij",
    ]


def test_chunk_text_applies_overlap():
    text = "abcdefghij"

    chunks = chunk_text(
        text = text,
        chunk_size = 4,
        overlap = 2,
    )

    assert chunks == [
        "abcd",
        "cdef",
        "efgh",
        "ghij",
    ]


def test_chunk_text_returns_empty_for_empty_text():
    chunks = chunk_text(
        text = "",
        chunk_size = 100,
        overlap = 20,
    )

    assert chunks == []


def test_chunk_text_returns_empty_for_whitespace_only_text():
    chunks = chunk_text(
        text = "   ",
        chunk_size = 100,
        overlap = 20,
    )

    assert chunks == []


def test_chunk_text_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            text = "some text",
            chunk_size = 0,
            overlap = 0,
        )


def test_chunk_text_rejects_negative_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            text = "some text",
            chunk_size = 100,
            overlap = -1,
        )


def test_chunk_text_rejects_overlap_equal_to_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            text = "some text",
            chunk_size = 100,
            overlap = 100,
        )


def test_chunk_text_rejects_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            text = "some text",
            chunk_size = 100,
            overlap = 101,
        )


def test_chunk_page_preserves_page_metadata():
    page = Page(
        document_name = "document.pdf",
        page_number = 7,
        text = "abcdefghij",
    )

    chunks = chunk_page(
        page = page,
        chunk_size = 4,
        overlap = 0,
        start_chunk_id = 10,
    )

    assert len(chunks) == 3

    assert chunks[0].chunk_id == 10
    assert chunks[0].document_name == "document.pdf"
    assert chunks[0].page_number == 7
    assert chunks[0].text == "abcd"

    assert chunks[1].chunk_id == 11
    assert chunks[1].document_name == "document.pdf"
    assert chunks[1].page_number == 7
    assert chunks[1].text == "efgh"

    assert chunks[2].chunk_id == 12
    assert chunks[2].document_name == "document.pdf"
    assert chunks[2].page_number == 7
    assert chunks[2].text == "ij"


def test_chunk_page_uses_requested_start_chunk_id():
    page = Page(
        document_name = "paper.pdf",
        page_number = 1,
        text = "abcdefgh",
    )

    chunks = chunk_page(
        page = page,
        chunk_size = 4,
        overlap = 0,
        start_chunk_id = 50,
    )

    assert [
        chunk.chunk_id
        for chunk in chunks
    ] == [
        50,
        51,
    ]


def test_chunk_pages_assigns_global_consecutive_chunk_ids():
    pages = [
        Page(
            document_name = "paper_a.pdf",
            page_number = 1,
            text = "abcdefgh",
        ),
        Page(
            document_name = "paper_a.pdf",
            page_number = 2,
            text = "ijklmnop",
        ),
    ]

    chunks = chunk_pages(
        pages = pages,
        chunk_size = 4,
        overlap = 0,
    )

    assert [
        chunk.chunk_id
        for chunk in chunks
    ] == [
        0,
        1,
        2,
        3,
    ]


def test_chunk_pages_preserves_document_and_page_information():
    pages = [
        Page(
            document_name = "paper_a.pdf",
            page_number = 1,
            text = "abcd",
        ),
        Page(
            document_name = "paper_b.pdf",
            page_number = 8,
            text = "efgh",
        ),
    ]

    chunks = chunk_pages(
        pages = pages,
        chunk_size = 10,
        overlap = 0,
    )

    assert len(chunks) == 2

    assert chunks[0].document_name == "paper_a.pdf"
    assert chunks[0].page_number == 1

    assert chunks[1].document_name == "paper_b.pdf"
    assert chunks[1].page_number == 8


def test_chunk_pages_skips_empty_pages_without_breaking_chunk_ids():
    pages = [
        Page(
            document_name = "paper.pdf",
            page_number = 1,
            text = "abcd",
        ),
        Page(
            document_name = "paper.pdf",
            page_number = 2,
            text = "   ",
        ),
        Page(
            document_name = "paper.pdf",
            page_number = 3,
            text = "efgh",
        ),
    ]

    chunks = chunk_pages(
        pages = pages,
        chunk_size = 10,
        overlap = 0,
    )

    assert len(chunks) == 2

    assert chunks[0].chunk_id == 0
    assert chunks[0].page_number == 1

    assert chunks[1].chunk_id == 1
    assert chunks[1].page_number == 3


def test_chunk_pages_returns_empty_for_empty_page_list():
    chunks = chunk_pages(
        pages = [],
        chunk_size = 100,
        overlap = 20,
    )

    assert chunks == []
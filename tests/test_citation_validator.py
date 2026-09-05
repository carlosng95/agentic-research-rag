from langchain_core.documents import Document

from agentic_research_rag.citation_validator import (
    extract_citation_numbers,
    validate_citations,
)


def make_documents(
    count: int,
) -> list[Document]:
    return [
        Document(
            page_content = (
                f"Evidence {index}"
            ),
            metadata = {
                "source": f"source-{index}",
            },
        )
        for index in range(
            1,
            count + 1,
        )
    ]


def test_extract_citation_numbers_preserves_order_and_removes_duplicates() -> None:
    citations = extract_citation_numbers(
        answer = (
            "First [SOURCE 3], then [SOURCE 1], "
            "and again [SOURCE 3]."
        )
    )

    assert citations == [
        3,
        1,
    ]


def test_validate_citations_accepts_valid_sources() -> None:
    result = validate_citations(
        answer = (
            "The evidence supports the statement "
            "[SOURCE 1] and provides additional "
            "context [SOURCE 3]."
        ),
        documents = make_documents(
            count = 3
        ),
    )

    assert result.valid is True
    assert result.has_citations is True

    assert result.cited_source_numbers == [
        1,
        3,
    ]

    assert result.invalid_source_numbers == []
    assert result.malformed_citations == []


def test_validate_citations_rejects_out_of_range_source() -> None:
    result = validate_citations(
        answer = (
            "The statement is supported "
            "[SOURCE 4]."
        ),
        documents = make_documents(
            count = 3
        ),
    )

    assert result.valid is False

    assert result.invalid_source_numbers == [
        4
    ]


def test_validate_citations_rejects_source_zero() -> None:
    result = validate_citations(
        answer = (
            "Invalid citation [SOURCE 0]."
        ),
        documents = make_documents(
            count = 3
        ),
    )

    assert result.valid is False

    assert result.invalid_source_numbers == [
        0
    ]


def test_validate_citations_detects_malformed_citation() -> None:
    result = validate_citations(
        answer = (
            "Malformed citation [SOURCE abc]."
        ),
        documents = make_documents(
            count = 3
        ),
    )

    assert result.valid is False

    assert result.malformed_citations == [
        "[SOURCE abc]"
    ]


def test_no_citations_is_structurally_valid() -> None:
    result = validate_citations(
        answer = (
            "No sufficient evidence was found."
        ),
        documents = [],
    )

    assert result.valid is True
    assert result.has_citations is False
    assert result.cited_source_numbers == []


def test_citation_is_invalid_when_no_documents_exist() -> None:
    result = validate_citations(
        answer = (
            "Unsupported citation [SOURCE 1]."
        ),
        documents = [],
    )

    assert result.valid is False

    assert result.invalid_source_numbers == [
        1
    ]
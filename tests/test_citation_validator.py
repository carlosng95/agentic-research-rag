import pytest

from agentic_research_rag.citation_validator import extract_citations, validate_citations


def test_extract_citations():
    answer = (
        "Semantic search uses dense representations [SOURCE 1]. "
        "Lexical retrieval uses term statistics [SOURCE 3]."
    )

    citations = extract_citations(answer)

    assert citations == [1, 3]


def test_extract_citations_is_case_insensitive():
    answer = "Evidence appears in [source 2] and [Source 4]."

    citations = extract_citations(answer)

    assert citations == [2, 4]


def test_validate_citations_accepts_valid_sources():
    answer = (
        "Semantic retrieval uses vector similarity [SOURCE 1] "
        "and lexical retrieval uses term matching [SOURCE 3]."
    )

    flags = validate_citations(
        answer = answer,
        source_count = 5,
    )

    assert flags == []


def test_validate_citations_detects_missing_citations():
    answer = "Hybrid retrieval combines multiple ranking strategies."

    flags = validate_citations(
        answer = answer,
        source_count = 5,
    )

    assert flags == ["missing_citations"]


def test_validate_citations_detects_invalid_source():
    answer = "Hybrid retrieval combines multiple ranking strategies [SOURCE 8]."

    flags = validate_citations(
        answer = answer,
        source_count = 5,
    )

    assert flags == ["invalid_source_8"]


def test_validate_citations_detects_multiple_invalid_sources():
    answer = (
        "First claim [SOURCE 8]. "
        "Second claim [SOURCE 10]. "
        "Third claim [SOURCE 8]."
    )

    flags = validate_citations(
        answer = answer,
        source_count = 5,
    )

    assert flags == [
        "invalid_source_8",
        "invalid_source_10",
    ]


def test_validate_citations_does_not_flag_missing_when_no_sources_exist():
    answer = "No evidence was available."

    flags = validate_citations(
        answer = answer,
        source_count = 0,
    )

    assert flags == []


def test_validate_citations_rejects_negative_source_count():
    with pytest.raises(
        ValueError,
        match = "source_count cannot be negative",
    ):
        validate_citations(
            answer = "Answer",
            source_count = -1,
        )

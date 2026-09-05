import re
from dataclasses import dataclass

from langchain_core.documents import Document


_CITATION_PATTERN = re.compile(
    r"\[SOURCE\s+(\d+)\]"
)

_SOURCE_TOKEN_PATTERN = re.compile(
    r"\[SOURCE[^\]]*\]"
)


@dataclass(frozen = True)
class CitationValidationResult:
    valid: bool
    has_citations: bool
    cited_source_numbers: list[int]
    invalid_source_numbers: list[int]
    malformed_citations: list[str]


def _unique_preserving_order(
    values: list[int],
) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def _unique_strings_preserving_order(
    values: list[str],
) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def extract_citation_numbers(
    answer: str,
) -> list[int]:
    citations = [
        int(match)
        for match in _CITATION_PATTERN.findall(
            answer
        )
    ]

    return _unique_preserving_order(
        values = citations
    )


def validate_citations(
    answer: str,
    documents: list[Document],
) -> CitationValidationResult:
    cited_source_numbers = extract_citation_numbers(
        answer = answer
    )

    invalid_source_numbers = [
        source_number
        for source_number in cited_source_numbers
        if (
            source_number < 1
            or source_number > len(documents)
        )
    ]

    source_tokens = _SOURCE_TOKEN_PATTERN.findall(
        answer
    )

    malformed_citations = [
        token
        for token in source_tokens
        if _CITATION_PATTERN.fullmatch(
            token
        ) is None
    ]

    malformed_citations = (
        _unique_strings_preserving_order(
            values = malformed_citations
        )
    )

    valid = (
        not invalid_source_numbers
        and not malformed_citations
    )

    return CitationValidationResult(
        valid = valid,
        has_citations = bool(
            cited_source_numbers
        ),
        cited_source_numbers = (
            cited_source_numbers
        ),
        invalid_source_numbers = (
            invalid_source_numbers
        ),
        malformed_citations = (
            malformed_citations
        ),
    )
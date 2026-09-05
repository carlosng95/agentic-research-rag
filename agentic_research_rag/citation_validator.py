import re


def extract_citations(answer: str) -> list[int]:
    """
    Extract SOURCE numbers cited in an answer.
    """

    matches = re.findall(r"\[SOURCE\s+(\d+)\]", answer, flags = re.IGNORECASE)

    return [int(match) for match in matches]


def validate_citations(answer: str, source_count: int) -> list[str]:
    """
    Validate citations against the available sources.

    Returns validation flags.
    """

    if source_count < 0:
        raise ValueError("source_count cannot be negative")

    citations = extract_citations(answer)
    flags: list[str] = []

    if source_count > 0 and not citations:
        flags.append("missing_citations")

    invalid_citations = sorted({
        citation
        for citation in citations
        if citation < 1 or citation > source_count
    })

    for citation in invalid_citations:
        flags.append(f"invalid_source_{citation}")

    return flags
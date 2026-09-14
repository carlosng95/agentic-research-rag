import pytest
from pydantic import ValidationError

from agentic_research_rag.serving.schemas import (
    HealthResponse,
    ReadyResponse,
    ResearchRequest,
)


def test_research_request() -> None:
    request = ResearchRequest(
        question = "What does the document say?",
        thread_id = "conversation-1",
    )

    assert request.question == "What does the document say?"
    assert request.thread_id == "conversation-1"


def test_research_request_strips_values() -> None:
    request = ResearchRequest(
        question = "  What does the document say?  ",
        thread_id = "  conversation-1  ",
    )

    assert request.question == "What does the document say?"
    assert request.thread_id == "conversation-1"


def test_research_request_uses_default_thread_id() -> None:
    request = ResearchRequest(
        question = "What does the document say?"
    )

    assert request.thread_id == "default"


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "     ",
    ],
)
def test_research_request_rejects_empty_question(
    question: str,
) -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(
            question = question
        )


@pytest.mark.parametrize(
    "thread_id",
    [
        "",
        " ",
        "     ",
    ],
)
def test_research_request_rejects_empty_thread_id(
    thread_id: str,
) -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(
            question = "Valid question",
            thread_id = thread_id,
        )


def test_health_response() -> None:
    response = HealthResponse()

    assert response.status == "healthy"


def test_ready_response() -> None:
    response = ReadyResponse(
        status = "ready"
    )

    assert response.status == "ready"
    assert response.detail is None
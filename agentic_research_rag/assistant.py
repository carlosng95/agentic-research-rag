import re
from typing import Any, Literal, Mapping

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from .bootstrap import build_application


class Source(BaseModel):
    source_number: int = Field(
        ge = 1
    )

    type: Literal[
        "paper",
        "web",
    ]

    ref: str
    locator: str | None = None
    snippet: str | None = None


class ResearchResponse(BaseModel):
    answer: str

    sources: list[Source] = Field(
        default_factory = list
    )

    cited_source_numbers: list[int] = Field(
        default_factory = list
    )

    tools_used: list[str] = Field(
        default_factory = list
    )

    flags: list[str] = Field(
        default_factory = list
    )

    paper_evidence_sufficient: bool = False
    citation_valid: bool = True


def _normalize_snippet(
    text: str,
    max_length: int = 300,
) -> str:
    normalized = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if len(normalized) <= max_length:
        return normalized

    return (
        normalized[:max_length].rstrip()
        + "..."
    )


def _document_to_source(
    document: Document,
    source_number: int,
) -> Source:
    url = document.metadata.get(
        "url"
    )

    if url:
        url = str(url)

        return Source(
            source_number = source_number,
            type = "web",
            ref = url,
            locator = url,
            snippet = _normalize_snippet(
                text = document.page_content
            ),
        )

    source = str(
        document.metadata.get(
            "source",
            "unknown",
        )
    )

    page_number = document.metadata.get(
        "page_number"
    )

    locator = None

    if page_number is not None:
        locator = f"page {page_number}"

    return Source(
        source_number = source_number,
        type = "paper",
        ref = source,
        locator = locator,
        snippet = _normalize_snippet(
            text = document.page_content
        ),
    )


def state_to_response(
    state: Mapping[str, Any],
) -> ResearchResponse:
    documents = state.get(
        "final_documents",
        [],
    )

    if not isinstance(
        documents,
        list,
    ):
        raise TypeError(
            "Expected final_documents to be a list."
        )

    sources = [
        _document_to_source(
            document = document,
            source_number = source_number,
        )
        for source_number, document in enumerate(
            documents,
            start = 1,
        )
    ]

    return ResearchResponse(
        answer = str(
            state.get(
                "final_answer",
                "",
            )
        ),
        sources = sources,
        cited_source_numbers = list(
            state.get(
                "cited_source_numbers",
                [],
            )
        ),
        tools_used = list(
            state.get(
                "tools_used",
                [],
            )
        ),
        flags = list(
            state.get(
                "flags",
                [],
            )
        ),
        paper_evidence_sufficient = bool(
            state.get(
                "sufficient",
                False,
            )
        ),
        citation_valid = bool(
            state.get(
                "citation_valid",
                True,
            )
        ),
    )


class ResearchAssistant:
    def __init__(
        self,
        graph: Runnable,
        thread_id: str = "default",
    ) -> None:
        if not thread_id.strip():
            raise ValueError(
                "thread_id cannot be empty."
            )

        self._graph = graph
        self._thread_id = thread_id

    @property
    def thread_id(
        self,
    ) -> str:
        return self._thread_id

    def ask(
        self,
        question: str,
    ) -> ResearchResponse:
        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        state = self._graph.invoke(
            {
                "question": question,
                "messages": [],
            },
            config = {
                "configurable": {
                    "thread_id": self._thread_id,
                }
            },
        )

        return state_to_response(
            state = state
        )


def build_assistant(
    thread_id: str = "default",
    **application_kwargs: Any,
) -> ResearchAssistant:
    graph = build_application(
        **application_kwargs
    )

    return ResearchAssistant(
        graph = graph,
        thread_id = thread_id,
    )
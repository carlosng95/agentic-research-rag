from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from agentic_research_rag.chains.sufficiency import (
    SufficiencyResult,
    build_sufficiency_chain,
)


class FakeStructuredChatModel:
    def __init__(
        self,
        result: SufficiencyResult,
    ) -> None:
        self.result = result
        self.calls = 0

    def with_structured_output(
        self,
        schema: type[SufficiencyResult],
    ) -> RunnableLambda:
        assert schema is SufficiencyResult

        def invoke_model(
            prompt: Any,
        ) -> SufficiencyResult:
            self.calls += 1
            return self.result

        return RunnableLambda(
            invoke_model
        )


def make_document() -> Document:
    return Document(
        page_content = (
            "Hybrid retrieval combines semantic "
            "and lexical retrieval."
        ),
        metadata = {
            "chunk_id": 0,
            "source": "retrieval.pdf",
            "page_number": 1,
        },
    )


def test_sufficiency_chain_returns_structured_result() -> None:
    model = FakeStructuredChatModel(
        result = SufficiencyResult(
            sufficient = True
        )
    )

    chain = build_sufficiency_chain(
        model = model
    )

    result = chain.invoke(
        {
            "question": "What is hybrid retrieval?",
            "answer": (
                "Hybrid retrieval combines semantic "
                "and lexical retrieval [SOURCE 1]."
            ),
            "documents": [
                make_document()
            ],
        }
    )

    assert isinstance(
        result,
        SufficiencyResult,
    )

    assert result.sufficient is True
    assert model.calls == 1


def test_sufficiency_chain_can_return_insufficient() -> None:
    model = FakeStructuredChatModel(
        result = SufficiencyResult(
            sufficient = False
        )
    )

    chain = build_sufficiency_chain(
        model = model
    )

    result = chain.invoke(
        {
            "question": "Unknown question",
            "answer": (
                "The evidence does not contain "
                "enough information."
            ),
            "documents": [
                make_document()
            ],
        }
    )

    assert result.sufficient is False
    assert model.calls == 1


def test_sufficiency_chain_skips_model_without_documents() -> None:
    model = FakeStructuredChatModel(
        result = SufficiencyResult(
            sufficient = True
        )
    )

    chain = build_sufficiency_chain(
        model = model
    )

    result = chain.invoke(
        {
            "question": "Unknown question",
            "answer": (
                "No relevant information was found "
                "in the available documents."
            ),
            "documents": [],
        }
    )

    assert result.sufficient is False
    assert model.calls == 0
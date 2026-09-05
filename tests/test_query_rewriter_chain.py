from typing import Any

from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from agentic_research_rag.chains.query_rewriter import (
    build_query_rewriter,
)


class CountingFakeListChatModel(FakeListChatModel):
    calls: int = 0

    def _call(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        self.calls += 1

        return super()._call(
            *args,
            **kwargs,
        )


def test_query_rewriter_returns_original_question_without_history() -> None:
    model = CountingFakeListChatModel(
        responses = [
            "This response should never be used."
        ]
    )

    chain = build_query_rewriter(
        model = model
    )

    result = chain.invoke(
        {
            "question": "What is hybrid retrieval?",
            "history": [],
        }
    )

    assert result == "What is hybrid retrieval?"
    assert model.calls == 0


def test_query_rewriter_rewrites_question_with_history() -> None:
    model = CountingFakeListChatModel(
        responses = [
            (
                "How does semantic retrieval differ "
                "from lexical retrieval?"
            )
        ]
    )

    chain = build_query_rewriter(
        model = model
    )

    result = chain.invoke(
        {
            "question": (
                "How does it differ from lexical retrieval?"
            ),
            "history": [
                HumanMessage(
                    content = (
                        "What is semantic retrieval?"
                    )
                ),
                AIMessage(
                    content = (
                        "Semantic retrieval represents "
                        "text using dense vectors."
                    )
                ),
            ],
        }
    )

    assert result == (
        "How does semantic retrieval differ "
        "from lexical retrieval?"
    )

    assert model.calls == 1


def test_query_rewriter_strips_model_output() -> None:
    model = CountingFakeListChatModel(
        responses = [
            (
                "\n"
                "How does hybrid retrieval combine "
                "semantic and lexical search?"
                "\n"
            )
        ]
    )

    chain = build_query_rewriter(
        model = model
    )

    result = chain.invoke(
        {
            "question": "How does it combine them?",
            "history": [
                HumanMessage(
                    content = (
                        "What is hybrid retrieval?"
                    )
                ),
                AIMessage(
                    content = (
                        "It combines semantic and "
                        "lexical retrieval."
                    )
                ),
            ],
        }
    )

    assert result == (
        "How does hybrid retrieval combine "
        "semantic and lexical search?"
    )

    assert model.calls == 1
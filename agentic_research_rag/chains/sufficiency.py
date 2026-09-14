from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
)
from pydantic import BaseModel

from .rag import format_documents
from ..observability.llm import with_llm_operation

class SufficiencyInput(TypedDict):
    question: str
    answer: str
    documents: list[Document]


class SufficiencyResult(BaseModel):
    sufficient: bool


_SUFFICIENCY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an evidence sufficiency evaluator.\n\n"
                "Determine whether the retrieved document evidence "
                "is sufficient to answer the user's question.\n\n"
                "Evidence is sufficient only when the question can be "
                "answered in a grounded way using the retrieved context "
                "without relying on unsupported external knowledge.\n\n"
                "If the evidence is incomplete, only tangentially related, "
                "or does not directly support an answer, mark it as "
                "insufficient."
            ),
        ),
        (
            "human",
            (
                "Question:\n"
                "{question}\n\n"
                "Current answer:\n"
                "{answer}\n\n"
                "Retrieved evidence:\n"
                "{evidence}"
            ),
        ),
    ]
)


def _prepare_input(
    state: SufficiencyInput,
) -> dict[str, str]:
    return {
        "question": state["question"],
        "answer": state["answer"],
        "evidence": format_documents(
            documents = state["documents"]
        ),
    }


def _no_documents(
    state: SufficiencyInput,
) -> SufficiencyResult:
    return SufficiencyResult(
        sufficient = False
    )


def build_sufficiency_chain(
    model: BaseChatModel,
) -> Runnable[SufficiencyInput, SufficiencyResult]:
    structured_model = with_llm_operation(
        model.with_structured_output(
            SufficiencyResult
        ),
        "sufficiency",
    )

    evaluation_chain = (
        RunnableLambda(
            _prepare_input
        )
        | _SUFFICIENCY_PROMPT
        | structured_model
    )

    return RunnableBranch(
        (
            lambda state: bool(
                state["documents"]
            ),
            evaluation_chain,
        ),
        RunnableLambda(
            _no_documents
        ),
    )
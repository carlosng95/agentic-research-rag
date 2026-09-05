from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)


class SynthesisInput(TypedDict):
    question: str
    paper_documents: list[Document]
    web_documents: list[Document]


class SynthesisResult(TypedDict):
    answer: str
    documents: list[Document]


_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a research synthesis assistant.\n\n"
                "Answer the user's question using only the provided "
                "evidence.\n\n"
                "Follow these rules:\n"
                "- Prefer local document evidence when it directly "
                "supports the answer.\n"
                "- Use web evidence to complement missing information.\n"
                "- Cite factual claims using [SOURCE N].\n"
                "- Never invent source numbers.\n"
                "- If sources disagree, explicitly describe the "
                "disagreement and cite the relevant sources.\n"
                "- Do not introduce unsupported external knowledge.\n"
                "- If the combined evidence is still insufficient, "
                "explicitly say so.\n"
                "- Keep the answer concise and evidence-focused."
            ),
        ),
        (
            "human",
            (
                "Question:\n"
                "{question}\n\n"
                "Combined evidence:\n"
                "{context}"
            ),
        ),
    ]
)


def format_synthesis_context(
    paper_documents: list[Document],
    web_documents: list[Document],
) -> str:
    parts: list[str] = []
    source_number = 1

    for document in paper_documents:
        source = document.metadata.get(
            "source",
            "unknown",
        )

        page_number = document.metadata.get(
            "page_number",
            "unknown",
        )

        chunk_id = document.metadata.get(
            "chunk_id",
            "unknown",
        )

        parts.append(
            (
                f"[SOURCE {source_number}]\n"
                f"Type: local document\n"
                f"Document: {source}\n"
                f"Page: {page_number}\n"
                f"Chunk ID: {chunk_id}\n\n"
                f"{document.page_content}"
            )
        )

        source_number += 1

    for document in web_documents:
        title = document.metadata.get(
            "title",
            document.metadata.get(
                "source",
                "unknown",
            ),
        )

        url = document.metadata.get(
            "url",
            document.metadata.get(
                "source",
                "unknown",
            ),
        )

        parts.append(
            (
                f"[SOURCE {source_number}]\n"
                f"Type: web\n"
                f"Title: {title}\n"
                f"URL: {url}\n\n"
                f"{document.page_content}"
            )
        )

        source_number += 1

    return "\n\n---\n\n".join(parts)


def _prepare_input(
    state: SynthesisInput,
) -> dict[str, object]:
    documents = (
        state["paper_documents"]
        + state["web_documents"]
    )

    return {
        "question": state["question"],
        "context": format_synthesis_context(
            paper_documents = state["paper_documents"],
            web_documents = state["web_documents"],
        ),
        "documents": documents,
    }


def _finalize_result(
    state: dict[str, object],
) -> SynthesisResult:
    answer = state["answer"]
    documents = state["documents"]

    if not isinstance(answer, str):
        raise TypeError(
            "Expected answer to be a string."
        )

    if not isinstance(documents, list):
        raise TypeError(
            "Expected documents to be a list."
        )

    return {
        "answer": answer,
        "documents": documents,
    }


def _no_evidence(
    state: dict[str, object],
) -> SynthesisResult:
    documents = state["documents"]

    if not isinstance(documents, list):
        raise TypeError(
            "Expected documents to be a list."
        )

    return {
        "answer": (
            "No sufficient evidence was found in the "
            "available documents or web sources."
        ),
        "documents": documents,
    }


def build_synthesis_chain(
    model: BaseChatModel,
) -> Runnable[SynthesisInput, SynthesisResult]:
    generation_chain = (
        RunnableLambda(
            lambda state: {
                "question": state["question"],
                "context": state["context"],
            }
        )
        | _SYNTHESIS_PROMPT
        | model
        | StrOutputParser()
    )

    generation_branch = (
        RunnablePassthrough.assign(
            answer = generation_chain
        )
        | RunnableLambda(
            _finalize_result
        )
    )

    return (
        RunnableLambda(
            _prepare_input
        )
        | RunnableBranch(
            (
                lambda state: bool(
                    state["documents"]
                ),
                generation_branch,
            ),
            RunnableLambda(
                _no_evidence
            ),
        )
    )
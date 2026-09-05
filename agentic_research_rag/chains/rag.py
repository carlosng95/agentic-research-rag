from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)


class RAGResult(TypedDict):
    answer: str
    documents: list[Document]


_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a research assistant answering questions "
                "using retrieved document evidence.\n\n"
                "Follow these rules:\n"
                "- Answer only using the provided context.\n"
                "- Do not use unsupported external knowledge.\n"
                "- Cite factual claims using [SOURCE N].\n"
                "- Never invent source numbers.\n"
                "- If the context does not contain enough information "
                "to answer the question, explicitly say so.\n"
                "- Keep the answer concise and evidence-focused."
            ),
        ),
        (
            "human",
            (
                "Question:\n"
                "{question}\n\n"
                "Context:\n"
                "{context}"
            ),
        ),
    ]
)


def format_documents(
    documents: list[Document],
) -> str:
    parts = []

    for source_number, document in enumerate(
        documents,
        start = 1,
    ):
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
                f"Document: {source}\n"
                f"Page: {page_number}\n"
                f"Chunk ID: {chunk_id}\n\n"
                f"{document.page_content}"
            )
        )

    return "\n\n---\n\n".join(parts)


def _retrieve(
    query: str,
    retriever: BaseRetriever,
) -> dict[str, object]:
    documents = retriever.invoke(
        query
    )

    return {
        "question": query,
        "documents": documents,
    }


def _prepare_generation_input(
    state: dict[str, object],
) -> dict[str, object]:
    documents = state["documents"]

    if not isinstance(documents, list):
        raise TypeError(
            "Expected documents to be a list."
        )

    return {
        **state,
        "context": format_documents(
            documents = documents
        ),
    }


def _finalize_result(
    state: dict[str, object],
) -> RAGResult:
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


def _no_results(
    state: dict[str, object],
) -> RAGResult:
    documents = state["documents"]

    if not isinstance(documents, list):
        raise TypeError(
            "Expected documents to be a list."
        )

    return {
        "answer": (
            "No relevant information was found "
            "in the available documents."
        ),
        "documents": documents,
    }


def build_rag_chain(
    retriever: BaseRetriever,
    model: BaseChatModel,
) -> Runnable[str, RAGResult]:
    generation_chain = (
        RunnableLambda(
            lambda state: {
                "question": state["question"],
                "context": state["context"],
            }
        )
        | _RAG_PROMPT
        | model
        | StrOutputParser()
    )

    generation_branch = (
        RunnableLambda(
            _prepare_generation_input
        )
        | RunnablePassthrough.assign(
            answer = generation_chain
        )
        | RunnableLambda(
            _finalize_result
        )
    )

    empty_branch = RunnableLambda(
        _no_results
    )

    return (
        RunnableLambda(
            lambda query: _retrieve(
                query = query,
                retriever = retriever,
            )
        )
        | RunnableBranch(
            (
                lambda state: bool(
                    state["documents"]
                ),
                generation_branch,
            ),
            empty_branch,
        )
    )
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from agentic_research_rag.retrieval.reranker import (
    RerankingRetriever,
)


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(
        self,
        query: str,
    ) -> list[Document]:
        return self.documents


class KeywordCrossEncoder(BaseCrossEncoder):
    def score(
        self,
        text_pairs: list[tuple[str, str]],
    ) -> list[float]:
        scores = []

        for query, document in text_pairs:
            query_terms = set(
                query.lower().split()
            )

            document_terms = set(
                document.lower().split()
            )

            scores.append(
                float(
                    len(
                        query_terms
                        & document_terms
                    )
                )
            )

        return scores


def make_document(
    chunk_id: int,
    text: str,
) -> Document:
    return Document(
        page_content = text,
        metadata = {
            "chunk_id": chunk_id,
            "source": "retrieval.pdf",
            "page_number": 1,
        },
    )


def test_reranker_orders_documents_by_cross_encoder_score() -> None:
    documents = [
        make_document(
            chunk_id = 0,
            text = "lexical matching",
        ),
        make_document(
            chunk_id = 1,
            text = "hybrid semantic retrieval",
        ),
        make_document(
            chunk_id = 2,
            text = "semantic retrieval",
        ),
    ]

    retriever = RerankingRetriever(
        base_retriever = StaticRetriever(
            documents = documents
        ),
        cross_encoder = KeywordCrossEncoder(),
        top_n = 3,
    )

    results = retriever.invoke(
        "semantic retrieval"
    )

    assert [
        document.metadata["chunk_id"]
        for document in results
    ] == [
        1,
        2,
        0,
    ]


def test_reranker_limits_results() -> None:
    documents = [
        make_document(
            chunk_id = index,
            text = f"document {index}",
        )
        for index in range(5)
    ]

    retriever = RerankingRetriever(
        base_retriever = StaticRetriever(
            documents = documents
        ),
        cross_encoder = KeywordCrossEncoder(),
        top_n = 2,
    )

    results = retriever.invoke(
        "document"
    )

    assert len(results) == 2


def test_reranker_handles_empty_retrieval() -> None:
    retriever = RerankingRetriever(
        base_retriever = StaticRetriever(
            documents = []
        ),
        cross_encoder = KeywordCrossEncoder(),
        top_n = 5,
    )

    results = retriever.invoke(
        "semantic retrieval"
    )

    assert results == []
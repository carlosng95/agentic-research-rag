from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from agentic_research_rag.retrieval.hybrid import (
    HybridRetriever,
    _tokenize,
)


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(
        self,
        query: str,
    ) -> list[Document]:
        return self.documents


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


def test_tokenize() -> None:
    tokens = _tokenize(
        "Hybrid Retrieval, BM25 + Semantic Search!"
    )

    assert tokens == [
        "hybrid",
        "retrieval",
        "bm25",
        "semantic",
        "search",
    ]


def test_hybrid_retriever_fuses_rankings() -> None:
    document_a = make_document(
        chunk_id = 0,
        text = "Document A",
    )

    document_b = make_document(
        chunk_id = 1,
        text = "Document B",
    )

    document_c = make_document(
        chunk_id = 2,
        text = "Document C",
    )

    document_d = make_document(
        chunk_id = 3,
        text = "Document D",
    )

    semantic_retriever = StaticRetriever(
        documents = [
            document_a,
            document_b,
            document_c,
        ]
    )

    bm25_retriever = StaticRetriever(
        documents = [
            document_b,
            document_a,
            document_d,
        ]
    )

    retriever = HybridRetriever(
        semantic_retriever = semantic_retriever,
        bm25_retriever = bm25_retriever,
        semantic_weight = 0.7,
        bm25_weight = 0.3,
        rrf_k = 60,
        top_k = 4,
    )

    results = retriever.invoke(
        "query"
    )

    assert [
        document.metadata["chunk_id"]
        for document in results
    ] == [
        0,
        1,
        2,
        3,
    ]


def test_hybrid_retriever_rewards_cross_retriever_overlap() -> None:
    document_a = make_document(
        chunk_id = 0,
        text = "Document A",
    )

    document_b = make_document(
        chunk_id = 1,
        text = "Document B",
    )

    document_c = make_document(
        chunk_id = 2,
        text = "Document C",
    )

    semantic_retriever = StaticRetriever(
        documents = [
            document_a,
            document_b,
        ]
    )

    bm25_retriever = StaticRetriever(
        documents = [
            document_c,
            document_b,
        ]
    )

    retriever = HybridRetriever(
        semantic_retriever = semantic_retriever,
        bm25_retriever = bm25_retriever,
        semantic_weight = 0.5,
        bm25_weight = 0.5,
        rrf_k = 60,
        top_k = 3,
    )

    results = retriever.invoke(
        "query"
    )

    assert (
        results[0].metadata["chunk_id"]
        == 1
    )


def test_duplicate_document_is_not_counted_twice_within_ranking() -> None:
    document_a = make_document(
        chunk_id = 0,
        text = "Document A",
    )

    document_b = make_document(
        chunk_id = 1,
        text = "Document B",
    )

    semantic_retriever = StaticRetriever(
        documents = [
            document_a,
            document_a,
            document_b,
        ]
    )

    bm25_retriever = StaticRetriever(
        documents = [
            document_b,
        ]
    )

    retriever = HybridRetriever(
        semantic_retriever = semantic_retriever,
        bm25_retriever = bm25_retriever,
        semantic_weight = 0.5,
        bm25_weight = 0.5,
        rrf_k = 60,
        top_k = 2,
    )

    results = retriever.invoke(
        "query"
    )

    assert len(results) == 2

    assert {
        document.metadata["chunk_id"]
        for document in results
    } == {
        0,
        1,
    }


def test_top_k_limits_results() -> None:
    documents = [
        make_document(
            chunk_id = index,
            text = f"Document {index}",
        )
        for index in range(5)
    ]

    retriever = HybridRetriever(
        semantic_retriever = StaticRetriever(
            documents = documents
        ),
        bm25_retriever = StaticRetriever(
            documents = documents
        ),
        top_k = 2,
    )

    results = retriever.invoke(
        "query"
    )

    assert len(results) == 2
import pytest

from agentic_research_rag.retrieval_pipeline import RetrievalPipeline
from agentic_research_rag.types import Chunk


def make_chunk(chunk_id: int) -> Chunk:
    return Chunk(
        chunk_id = chunk_id,
        document_name = f"paper_{chunk_id}.pdf",
        page_number = 1,
        text = f"Chunk {chunk_id}",
    )


class FakeRetriever:
    def __init__(self, results: list[Chunk]) -> None:
        self._results = results
        self.calls: list[dict] = []

    def retrieve(self, query: str, k: int, candidate_k: int) -> list[Chunk]:
        self.calls.append(
            {
                "query": query,
                "k": k,
                "candidate_k": candidate_k,
            }
        )

        return self._results[:k]


class FakeReranker:
    def __init__(self, results: list[Chunk]) -> None:
        self._results = results
        self.calls: list[dict] = []

    def rerank(self, query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
        self.calls.append(
            {
                "query": query,
                "chunks": chunks,
                "k": k,
            }
        )

        return self._results[:k]


def test_retrieval_pipeline_passes_correct_parameters():
    chunks = [
        make_chunk(1),
        make_chunk(2),
        make_chunk(3),
    ]

    retriever = FakeRetriever(
        results = chunks,
    )

    reranker = FakeReranker(
        results = chunks,
    )

    pipeline = RetrievalPipeline(
        retriever = retriever,
        reranker = reranker,
        candidate_k = 30,
        rerank_k = 20,
        final_k = 5,
    )

    query = "How does hybrid retrieval work?"

    pipeline.search(
        query = query,
    )

    assert len(retriever.calls) == 1
    assert retriever.calls[0]["query"] == query
    assert retriever.calls[0]["k"] == 20
    assert retriever.calls[0]["candidate_k"] == 30


def test_retrieval_pipeline_passes_candidates_to_reranker():
    chunks = [
        make_chunk(1),
        make_chunk(2),
        make_chunk(3),
    ]

    retriever = FakeRetriever(
        results = chunks,
    )

    reranker = FakeReranker(
        results = chunks,
    )

    pipeline = RetrievalPipeline(
        retriever = retriever,
        reranker = reranker,
        candidate_k = 30,
        rerank_k = 20,
        final_k = 2,
    )

    query = "What is semantic search?"

    pipeline.search(
        query = query,
    )

    assert len(reranker.calls) == 1
    assert reranker.calls[0]["query"] == query
    assert reranker.calls[0]["chunks"] == chunks
    assert reranker.calls[0]["k"] == 2


def test_retrieval_pipeline_returns_reranked_results():
    retrieved_chunks = [
        make_chunk(1),
        make_chunk(2),
        make_chunk(3),
    ]

    reranked_chunks = [
        retrieved_chunks[2],
        retrieved_chunks[0],
        retrieved_chunks[1],
    ]

    retriever = FakeRetriever(
        results = retrieved_chunks,
    )

    reranker = FakeReranker(
        results = reranked_chunks,
    )

    pipeline = RetrievalPipeline(
        retriever = retriever,
        reranker = reranker,
        candidate_k = 30,
        rerank_k = 20,
        final_k = 2,
    )

    results = pipeline.search(
        query = "What is semantic search?"
    )

    assert len(results) == 2
    assert results[0].chunk_id == 3
    assert results[1].chunk_id == 1


def test_retrieval_pipeline_returns_empty_for_empty_query():
    chunks = [
        make_chunk(1),
    ]

    retriever = FakeRetriever(
        results = chunks,
    )

    reranker = FakeReranker(
        results = chunks,
    )

    pipeline = RetrievalPipeline(
        retriever = retriever,
        reranker = reranker,
        candidate_k = 30,
        rerank_k = 20,
        final_k = 5,
    )

    results = pipeline.search(
        query = "   "
    )

    assert results == []
    assert retriever.calls == []
    assert reranker.calls == []


def test_retrieval_pipeline_rejects_zero_candidate_k():
    with pytest.raises(
        ValueError,
        match = "candidate_k must be greater than 0",
    ):
        RetrievalPipeline(
            retriever = FakeRetriever([]),
            reranker = FakeReranker([]),
            candidate_k = 0,
            rerank_k = 20,
            final_k = 5,
        )


def test_retrieval_pipeline_rejects_zero_rerank_k():
    with pytest.raises(
        ValueError,
        match = "rerank_k must be greater than 0",
    ):
        RetrievalPipeline(
            retriever = FakeRetriever([]),
            reranker = FakeReranker([]),
            candidate_k = 30,
            rerank_k = 0,
            final_k = 5,
        )


def test_retrieval_pipeline_rejects_zero_final_k():
    with pytest.raises(
        ValueError,
        match = "final_k must be greater than 0",
    ):
        RetrievalPipeline(
            retriever = FakeRetriever([]),
            reranker = FakeReranker([]),
            candidate_k = 30,
            rerank_k = 20,
            final_k = 0,
        )


def test_retrieval_pipeline_rejects_rerank_k_greater_than_candidate_k():
    with pytest.raises(
        ValueError,
        match = "rerank_k cannot be greater than candidate_k",
    ):
        RetrievalPipeline(
            retriever = FakeRetriever([]),
            reranker = FakeReranker([]),
            candidate_k = 10,
            rerank_k = 20,
            final_k = 5,
        )


def test_retrieval_pipeline_rejects_final_k_greater_than_rerank_k():
    with pytest.raises(
        ValueError,
        match = "final_k cannot be greater than rerank_k",
    ):
        RetrievalPipeline(
            retriever = FakeRetriever([]),
            reranker = FakeReranker([]),
            candidate_k = 30,
            rerank_k = 10,
            final_k = 20,
        )

import pytest

from agentic_research_rag.reranking.cross_encoder import CrossEncoderReranker
from agentic_research_rag.types import Chunk


class FakeCrossEncoderModel:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]):
        self.calls.append(pairs)
        return self._scores


def make_chunk(chunk_id: int, text: str) -> Chunk:
    return Chunk(
        chunk_id = chunk_id,
        document_name = f"paper_{chunk_id}.pdf",
        page_number = 1,
        text = text,
    )


def make_reranker(scores: list[float]) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker._model = FakeCrossEncoderModel(scores)
    reranker._model_name = "fake-cross-encoder"

    return reranker


def test_cross_encoder_builds_query_chunk_pairs():
    chunks = [
        make_chunk(1, "First evidence"),
        make_chunk(2, "Second evidence"),
    ]

    reranker = make_reranker(
        scores = [0.8, 0.2],
    )

    query = "What is semantic retrieval?"

    reranker.rerank(
        query = query,
        chunks = chunks,
        k = 2,
    )

    assert reranker._model.calls == [
        [
            (query, "First evidence"),
            (query, "Second evidence"),
        ]
    ]


def test_cross_encoder_orders_chunks_by_score():
    chunks = [
        make_chunk(1, "First evidence"),
        make_chunk(2, "Second evidence"),
        make_chunk(3, "Third evidence"),
    ]

    reranker = make_reranker(
        scores = [0.2, 0.9, 0.5],
    )

    results = reranker.rerank(
        query = "What is semantic retrieval?",
        chunks = chunks,
        k = 3,
    )

    assert results[0].chunk_id == 2
    assert results[1].chunk_id == 3
    assert results[2].chunk_id == 1


def test_cross_encoder_assigns_scores_to_results():
    chunks = [
        make_chunk(1, "First evidence"),
        make_chunk(2, "Second evidence"),
    ]

    reranker = make_reranker(
        scores = [0.3, 0.8],
    )

    results = reranker.rerank(
        query = "What is semantic retrieval?",
        chunks = chunks,
        k = 2,
    )

    assert results[0].chunk_id == 2
    assert results[0].score == pytest.approx(0.8)

    assert results[1].chunk_id == 1
    assert results[1].score == pytest.approx(0.3)


def test_cross_encoder_limits_results_to_k():
    chunks = [
        make_chunk(1, "First evidence"),
        make_chunk(2, "Second evidence"),
        make_chunk(3, "Third evidence"),
    ]

    reranker = make_reranker(
        scores = [0.2, 0.9, 0.5],
    )

    results = reranker.rerank(
        query = "What is semantic retrieval?",
        chunks = chunks,
        k = 2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == 2
    assert results[1].chunk_id == 3


def test_cross_encoder_does_not_modify_original_chunks():
    chunks = [
        make_chunk(1, "First evidence"),
        make_chunk(2, "Second evidence"),
    ]

    reranker = make_reranker(
        scores = [0.2, 0.9],
    )

    results = reranker.rerank(
        query = "What is semantic retrieval?",
        chunks = chunks,
        k = 2,
    )

    assert chunks[0].score is None
    assert chunks[1].score is None

    assert results[0].score is not None
    assert results[1].score is not None


def test_cross_encoder_returns_empty_for_empty_chunks():
    reranker = make_reranker(
        scores = [],
    )

    results = reranker.rerank(
        query = "What is semantic retrieval?",
        chunks = [],
        k = 5,
    )

    assert results == []
    assert reranker._model.calls == []


def test_cross_encoder_returns_empty_for_empty_query():
    chunks = [
        make_chunk(1, "Evidence"),
    ]

    reranker = make_reranker(
        scores = [0.5],
    )

    results = reranker.rerank(
        query = "   ",
        chunks = chunks,
        k = 5,
    )

    assert results == []
    assert reranker._model.calls == []


def test_cross_encoder_returns_empty_for_zero_k():
    chunks = [
        make_chunk(1, "Evidence"),
    ]

    reranker = make_reranker(
        scores = [0.5],
    )

    results = reranker.rerank(
        query = "What is semantic retrieval?",
        chunks = chunks,
        k = 0,
    )

    assert results == []
    assert reranker._model.calls == []


def test_cross_encoder_exposes_model_id():
    reranker = make_reranker(
        scores = [],
    )

    assert reranker.model_id == "fake-cross-encoder"

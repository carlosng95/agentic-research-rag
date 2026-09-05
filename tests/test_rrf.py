import pytest

from agentic_research_rag.fusion.rrf import reciprocal_rank_fusion
from agentic_research_rag.types import Chunk


def make_chunk(chunk_id: int) -> Chunk:
    return Chunk(
        chunk_id = chunk_id,
        document_name = f"paper_{chunk_id}.pdf",
        page_number = 1,
        text = f"Chunk {chunk_id}",
    )


def test_rrf_rewards_chunks_present_high_in_multiple_rankings():
    chunk_1 = make_chunk(1)
    chunk_2 = make_chunk(2)
    chunk_3 = make_chunk(3)
    chunk_4 = make_chunk(4)

    ranking_a = [chunk_1, chunk_2, chunk_3]
    ranking_b = [chunk_4, chunk_1, chunk_3]

    results = reciprocal_rank_fusion(
        rankings = [ranking_a, ranking_b],
        rrf_k = 60,
        top_n = 4,
    )

    assert results[0].chunk_id == 1


def test_rrf_score_is_calculated_correctly():
    chunk_1 = make_chunk(1)
    chunk_2 = make_chunk(2)

    ranking_a = [chunk_1, chunk_2]
    ranking_b = [chunk_2, chunk_1]

    results = reciprocal_rank_fusion(
        rankings = [ranking_a, ranking_b],
        rrf_k = 60,
        top_n = 2,
    )

    expected_score = (1 / 61) + (1 / 62)

    assert results[0].chunk_id == 1
    assert results[0].score == pytest.approx(expected_score)


def test_rrf_ignores_duplicate_chunk_within_same_ranking():
    chunk_1 = make_chunk(1)
    chunk_2 = make_chunk(2)

    ranking = [chunk_1, chunk_1, chunk_2]

    results = reciprocal_rank_fusion(
        rankings = [ranking],
        rrf_k = 60,
        top_n = 2,
    )

    assert results[0].chunk_id == 1
    assert results[0].score == pytest.approx(1 / 61)


def test_rrf_returns_empty_for_no_rankings():
    results = reciprocal_rank_fusion(
        rankings = [],
        rrf_k = 60,
        top_n = 5,
    )

    assert results == []


def test_rrf_returns_empty_when_top_n_is_zero():
    chunk_1 = make_chunk(1)

    results = reciprocal_rank_fusion(
        rankings = [[chunk_1]],
        rrf_k = 60,
        top_n = 0,
    )

    assert results == []


def test_rrf_rejects_negative_rrf_k():
    with pytest.raises(
        ValueError,
        match = "rrf_k cannot be negative",
    ):
        reciprocal_rank_fusion(
            rankings = [],
            rrf_k = -1,
            top_n = 5,
        )
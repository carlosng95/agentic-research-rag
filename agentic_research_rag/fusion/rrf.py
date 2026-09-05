from dataclasses import replace

from ..types import Chunk

def reciprocal_rank_fusion(
    rankings: list[list[Chunk]],
    rrf_k: int = 60,
    top_n: int = 5,
) -> list[Chunk]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    RRF ignores the original retrieval scores and uses only
    the position of each chunk in each ranking.
    """

    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative")

    if top_n <= 0:
        return []

    if not rankings:
        return []

    rrf_scores: dict[int, float] = {}
    chunks_by_id: dict[int, Chunk] = {}

    for ranking in rankings:
        seen_chunk_ids: set[int] = set()

        for rank, chunk in enumerate(ranking, start = 1):
            if chunk.chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk.chunk_id)

            score = 1.0 / (rrf_k + rank)

            rrf_scores[chunk.chunk_id] = (
                rrf_scores.get(chunk.chunk_id, 0.0) + score
            )

            chunks_by_id[chunk.chunk_id] = chunk

    sorted_chunk_ids = sorted(
        rrf_scores,
        key = lambda chunk_id: (-rrf_scores[chunk_id], chunk_id),
    )

    results: list[Chunk] = []

    for chunk_id in sorted_chunk_ids[:top_n]:
        chunk = chunks_by_id[chunk_id]

        result = replace(
            chunk,
            score = rrf_scores[chunk_id],
        )

        results.append(result)

    return results
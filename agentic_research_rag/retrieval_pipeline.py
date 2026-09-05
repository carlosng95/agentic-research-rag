from .reranking.base import Reranker
from .retriever import Retriever
from .types import Chunk


class RetrievalPipeline:
    """
    Coordinate hybrid retrieval and reranking.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        candidate_k: int,
        rerank_k: int,
        final_k: int,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")

        if rerank_k <= 0:
            raise ValueError("rerank_k must be greater than 0")

        if final_k <= 0:
            raise ValueError("final_k must be greater than 0")

        if rerank_k > candidate_k:
            raise ValueError("rerank_k cannot be greater than candidate_k")

        if final_k > rerank_k:
            raise ValueError("final_k cannot be greater than rerank_k")

        self._retriever = retriever
        self._reranker = reranker
        self._candidate_k = candidate_k
        self._rerank_k = rerank_k
        self._final_k = final_k

    def search(self, query: str) -> list[Chunk]:
        if not query.strip():
            return []

        candidates = self._retriever.retrieve(
            query = query,
            k = self._rerank_k,
            candidate_k = self._candidate_k,
        )

        return self._reranker.rerank(
            query = query,
            chunks = candidates,
            k = self._final_k,
        )
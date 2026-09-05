from .indexing.bm25_index import BM25Index
from .indexing.semantic_index import SemanticIndex
from .types import Chunk
from .fusion.rrf import reciprocal_rank_fusion

class Retriever:
    """
    Coordinate retrieval strategies over the corpus.
    """

    def __init__(self, semantic_index: SemanticIndex, bm25_index: BM25Index, rrf_k: int = 60) -> None:
        self._semantic_index = semantic_index
        self._bm25_index = bm25_index
        self._rrf_k = rrf_k
        

    def semantic_search(self, query: str, k: int = 5) -> list[Chunk]:
        """
        Search using semantic similarity.
        """
        return self._semantic_index.search(query = query, k = k)

    def bm25_search(self, query: str, k: int = 5) -> list[Chunk]:
        """
        Search using BM25 lexical retrieval.
        """
        return self._bm25_index.search(query = query, k = k)
    
    def retrieve(self, query: str, k: int = 5, candidate_k: int = 20) -> list[Chunk]:
        if k <= 0:
            return []

        if candidate_k <= 0:
            return []

        semantic_results = self.semantic_search(query = query, k = candidate_k)
        bm25_results = self.bm25_search(query = query, k = candidate_k)

        return reciprocal_rank_fusion(
            rankings = [semantic_results, bm25_results],
            rrf_k = self._rrf_k,
            top_n = k,
        )
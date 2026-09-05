from abc import ABC, abstractmethod
from ..types import Chunk

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], k: int = 5) -> list[Chunk]:
        raise NotImplementedError
    
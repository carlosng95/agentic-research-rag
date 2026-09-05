from dataclasses import replace

import numpy as np

from ..config import get_required_env
from ..types import Chunk
from .base import Reranker


class CrossEncoderReranker(Reranker):
    """
    Rerank chunks using a local cross-encoder model.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self._model_name = model_name or get_required_env("CROSS_ENCODER_MODEL")
        self._model = CrossEncoder(self._model_name)

    @property
    def model_id(self) -> str:
        return self._model_name

    def rerank(self, query: str, chunks: list[Chunk], k: int = 5) -> list[Chunk]:
        if k <= 0:
            return []

        if not query.strip():
            return []

        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]

        scores = self._model.predict(pairs)
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[Chunk] = []

        for index in top_indices:
            chunk = chunks[int(index)]

            result = replace(
                chunk,
                score = float(scores[index]),
            )

            results.append(result)

        return results
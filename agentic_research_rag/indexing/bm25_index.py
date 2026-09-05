from dataclasses import replace

import numpy as np
from rank_bm25 import BM25Okapi

from ..tokenizer import Tokenizer
from ..types import Chunk


class BM25Index:
    """
    Lexical search index based on BM25.
    """

    def __init__(self, chunks: list[Chunk], tokenizer: Tokenizer) -> None:
        self._chunks = chunks
        self._tokenizer = tokenizer

        tokenized_corpus = [
            self._tokenizer.tokenize(chunk.text) for chunk in self._chunks
        ]

        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """
        Return the k chunks with the highest BM25 scores.
        """

        if k <= 0:
            return []

        if not query.strip():
            return []

        query_tokens = self._tokenizer.tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:k]

        results: list[Chunk] = []

        for index in top_indices:
            chunk = self._chunks[int(index)]

            result = replace(
                chunk,
                score=float(scores[index]),
            )

            results.append(result)

        return results
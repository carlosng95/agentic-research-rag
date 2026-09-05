from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np

from ..providers.embeddings import EmbeddingProvider
from ..types import Chunk


class SemanticIndexMismatchError(ValueError):
    """
    Raised when a persisted semantic index does not match
    the current corpus or embedding model.
    """

    pass


class SemanticIndex:
    """
    Semantic search index built from document chunks.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider,
        embedding_matrix: np.ndarray,
    ) -> None:

        self._chunks = chunks
        self._embedding_provider = embedding_provider
        self._embedding_matrix = embedding_matrix

    @staticmethod
    def _corpus_fingerprint(
        chunks: list[Chunk],
    ) -> str:
        """
        Generate a fingerprint representing the corpus.

        If any chunk changes, the fingerprint changes.
        """

        hasher = hashlib.sha256()

        for chunk in chunks:
            content = (
                f"{chunk.chunk_id}|"
                f"{chunk.document_name}|"
                f"{chunk.page_number}|"
                f"{chunk.text}"
            )

            hasher.update(
                content.encode("utf-8")
            )

        return hasher.hexdigest()

    @staticmethod
    def _normalize_matrix(
        matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize embedding vectors to unit length.
        """

        norms = np.linalg.norm(
            matrix,
            axis=1,
            keepdims=True,
        )

        norms = np.where(
            norms == 0,
            1.0,
            norms,
        )

        return matrix / norms

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider,
    ) -> "SemanticIndex":
        """
        Build a semantic index by generating embeddings
        for all chunks.
        """

        if not chunks:
            raise ValueError(
                "Cannot build semantic index "
                "without chunks."
            )

        if not embedding_provider.available:
            raise ValueError(
                "Embedding provider is not available."
            )

        texts = [chunk.text for chunk in chunks]

        vectors = embedding_provider.embed(texts)

        if len(vectors) != len(chunks):
            raise ValueError(
                "Number of embeddings does not match "
                "number of chunks."
            )

        matrix = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if matrix.ndim != 2:
            raise ValueError(
                "Embedding matrix must be two-dimensional."
            )

        matrix = cls._normalize_matrix(
            matrix
        )

        return cls(
            chunks=chunks,
            embedding_provider=embedding_provider,
            embedding_matrix=matrix,
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save the semantic index to disk.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fingerprint = self._corpus_fingerprint(
            self._chunks
        )

        np.savez_compressed(
            path,
            embeddings=self._embedding_matrix,
            corpus_fingerprint=fingerprint,
            model_id=self._embedding_provider.model_id,
            chunk_count=len(self._chunks),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider,
    ) -> "SemanticIndex":
        """
        Load an existing semantic index from disk.

        The stored corpus and embedding model are validated
        before the index is accepted.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Semantic index not found: {path}"
            )

        data = np.load(
            path,
            allow_pickle=False,
        )

        matrix = data["embeddings"]

        stored_fingerprint = str(
            data["corpus_fingerprint"].item()
        )

        stored_model_id = str(
            data["model_id"].item()
        )

        stored_chunk_count = int(
            data["chunk_count"].item()
        )

        current_fingerprint = (
            cls._corpus_fingerprint(
                chunks
            )
        )

        if stored_fingerprint != current_fingerprint:
            raise SemanticIndexMismatchError(
                "Semantic index does not match the current corpus."
            )

        if stored_model_id != embedding_provider.model_id:
            raise SemanticIndexMismatchError(
                "Semantic index was created with "
                f"'{stored_model_id}', but the current provider uses "
                f"'{embedding_provider.model_id}'."
            )

        if stored_chunk_count != len(chunks):
            raise SemanticIndexMismatchError(
                "Semantic index chunk count does not match the current corpus."
            )

        return cls(
            chunks=chunks,
            embedding_provider=embedding_provider,
            embedding_matrix=matrix,
        )

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[Chunk]:
        """
        Return the k chunks most semantically similar
        to the query.
        """

        if k <= 0:
            return []

        if not query.strip():
            return []

        query_vectors = self._embedding_provider.embed([query])

        if not query_vectors:
            return []

        query_vector = np.asarray(
            query_vectors[0],
            dtype=np.float32,
        )

        query_norm = np.linalg.norm(
            query_vector
        )

        if query_norm == 0:
            return []

        query_vector = (
            query_vector / query_norm
        )

        if (
            query_vector.shape[0]
            != self._embedding_matrix.shape[1]
        ):
            raise ValueError(
                "Query embedding dimension does not "
                "match semantic index dimension."
            )

        scores = (
            self._embedding_matrix
            @ query_vector
        )

        top_indices = np.argsort(
            scores
        )[::-1][:k]

        results: list[Chunk] = []

        for index in top_indices:
            chunk = self._chunks[int(index)]

            result = replace(
                chunk,
                score=float(scores[index]),
            )

            results.append(result)

        return results
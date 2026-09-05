import pytest

from agentic_research_rag.indexing.semantic_index import (
    SemanticIndex,
    SemanticIndexMismatchError,
)
from agentic_research_rag.providers.embeddings import EmbeddingProvider
from agentic_research_rag.types import Chunk


class FakeEmbeddingProvider(EmbeddingProvider):
    available = True

    def __init__(
        self,
        embeddings: dict[str, list[float]],
        model_id: str = "fake-embedding-model",
    ) -> None:
        self._embeddings = embeddings
        self._model_id = model_id
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)

        return [
            self._embeddings[text]
            for text in texts
        ]


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id = 1,
            document_name = "paper.pdf",
            page_number = 1,
            text = "semantic search",
        ),
        Chunk(
            chunk_id = 2,
            document_name = "paper.pdf",
            page_number = 2,
            text = "vector retrieval",
        ),
        Chunk(
            chunk_id = 3,
            document_name = "paper.pdf",
            page_number = 3,
            text = "weather forecast",
        ),
    ]


def make_provider(
    model_id: str = "fake-embedding-model",
) -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(
        embeddings = {
            "semantic search": [1.0, 0.0],
            "vector retrieval": [0.8, 0.2],
            "weather forecast": [0.0, 1.0],
            "search query": [1.0, 0.0],
            "weather query": [0.0, 1.0],
        },
        model_id = model_id,
    )


def test_semantic_index_build_embeds_all_chunks():
    chunks = make_chunks()
    provider = make_provider()

    SemanticIndex.build(
        chunks = chunks,
        embedding_provider = provider,
    )

    assert len(provider.calls) == 1

    assert provider.calls[0] == [
        "semantic search",
        "vector retrieval",
        "weather forecast",
    ]


def test_semantic_search_returns_most_similar_chunk():
    chunks = make_chunks()
    provider = make_provider()

    index = SemanticIndex.build(
        chunks = chunks,
        embedding_provider = provider,
    )

    results = index.search(
        query = "search query",
        k = 2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == 1
    assert results[1].chunk_id == 2


def test_semantic_search_returns_similarity_score():
    chunks = make_chunks()
    provider = make_provider()

    index = SemanticIndex.build(
        chunks = chunks,
        embedding_provider = provider,
    )

    results = index.search(
        query = "search query",
        k = 1,
    )

    assert results[0].score == pytest.approx(1.0)


def test_semantic_search_normalizes_embeddings():
    chunks = [
        Chunk(
            chunk_id = 1,
            document_name = "paper.pdf",
            page_number = 1,
            text = "document",
        ),
    ]

    provider = FakeEmbeddingProvider(
        embeddings = {
            "document": [10.0, 0.0],
            "query": [5.0, 0.0],
        },
    )

    index = SemanticIndex.build(
        chunks = chunks,
        embedding_provider = provider,
    )

    results = index.search(
        query = "query",
        k = 1,
    )

    assert results[0].score == pytest.approx(1.0)


def test_semantic_search_returns_empty_for_empty_query():
    chunks = make_chunks()
    provider = make_provider()

    index = SemanticIndex.build(
        chunks = chunks,
        embedding_provider = provider,
    )

    calls_before_search = len(provider.calls)

    results = index.search(
        query = "   ",
        k = 5,
    )

    assert results == []
    assert len(provider.calls) == calls_before_search


def test_semantic_search_returns_empty_for_zero_k():
    chunks = make_chunks()
    provider = make_provider()

    index = SemanticIndex.build(
        chunks = chunks,
        embedding_provider = provider,
    )

    calls_before_search = len(provider.calls)

    results = index.search(
        query = "search query",
        k = 0,
    )

    assert results == []
    assert len(provider.calls) == calls_before_search


def test_semantic_index_rejects_empty_chunks():
    provider = make_provider()

    with pytest.raises(
        ValueError,
        match = "Cannot build semantic index without chunks",
    ):
        SemanticIndex.build(
            chunks = [],
            embedding_provider = provider,
        )


def test_semantic_index_save_and_load(tmp_path):
    chunks = make_chunks()
    provider = make_provider()

    index = SemanticIndex.build(
        chunks = chunks,
        embedding_provider = provider,
    )

    path = tmp_path / "semantic_index.npz"

    index.save(path)

    assert path.exists()

    calls_after_build = len(provider.calls)

    loaded_index = SemanticIndex.load(
        path = path,
        chunks = chunks,
        embedding_provider = provider,
    )

    assert len(provider.calls) == calls_after_build

    results = loaded_index.search(
        query = "weather query",
        k = 1,
    )

    assert results[0].chunk_id == 3


def test_semantic_index_load_rejects_different_corpus(tmp_path):
    chunks = make_chunks()
    provider = make_provider()

    index = SemanticIndex.build(
        chunks = chunks,
        embedding_provider = provider,
    )

    path = tmp_path / "semantic_index.npz"
    index.save(path)

    modified_chunks = make_chunks()
    modified_chunks[0].text = "completely different text"

    with pytest.raises(
        SemanticIndexMismatchError,
        match = "does not match the current corpus",
    ):
        SemanticIndex.load(
            path = path,
            chunks = modified_chunks,
            embedding_provider = provider,
        )


def test_semantic_index_load_rejects_different_model(tmp_path):
    chunks = make_chunks()

    provider = make_provider(
        model_id = "model-a",
    )

    index = SemanticIndex.build(
        chunks = chunks,
        embedding_provider = provider,
    )

    path = tmp_path / "semantic_index.npz"
    index.save(path)

    different_provider = make_provider(
        model_id = "model-b",
    )

    with pytest.raises(
        SemanticIndexMismatchError,
        match = "model-a",
    ):
        SemanticIndex.load(
            path = path,
            chunks = chunks,
            embedding_provider = different_provider,
        )


def test_semantic_index_load_rejects_missing_file(tmp_path):
    provider = make_provider()

    path = tmp_path / "missing_index.npz"

    with pytest.raises(
        FileNotFoundError,
        match = "Semantic index not found",
    ):
        SemanticIndex.load(
            path = path,
            chunks = make_chunks(),
            embedding_provider = provider,
        )

from agentic_research_rag.indexing.semantic_index_manager import load_or_build_semantic_index
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
    ]


def make_provider(
    model_id: str = "fake-embedding-model",
) -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(
        embeddings = {
            "semantic search": [1.0, 0.0],
            "vector retrieval": [0.8, 0.2],
            "modified document text": [0.5, 0.5],
            "query": [1.0, 0.0],
        },
        model_id = model_id,
    )


def test_manager_builds_and_saves_index_when_file_does_not_exist(tmp_path):
    chunks = make_chunks()
    provider = make_provider()

    path = tmp_path / "semantic_index.npz"

    assert not path.exists()

    index = load_or_build_semantic_index(
        path = path,
        chunks = chunks,
        embedding_provider = provider,
    )

    assert path.exists()

    assert provider.calls == [
        [
            "semantic search",
            "vector retrieval",
        ]
    ]

    results = index.search(
        query = "query",
        k = 1,
    )

    assert results[0].chunk_id == 1


def test_manager_loads_existing_valid_index_without_reembedding_corpus(tmp_path):
    chunks = make_chunks()
    provider = make_provider()

    path = tmp_path / "semantic_index.npz"

    load_or_build_semantic_index(
        path = path,
        chunks = chunks,
        embedding_provider = provider,
    )

    assert len(provider.calls) == 1

    second_provider = make_provider()

    index = load_or_build_semantic_index(
        path = path,
        chunks = chunks,
        embedding_provider = second_provider,
    )

    assert second_provider.calls == []

    results = index.search(
        query = "query",
        k = 1,
    )

    assert results[0].chunk_id == 1

    assert second_provider.calls == [
        ["query"],
    ]


def test_manager_rebuilds_index_when_corpus_changes(tmp_path):
    chunks = make_chunks()
    provider = make_provider()

    path = tmp_path / "semantic_index.npz"

    load_or_build_semantic_index(
        path = path,
        chunks = chunks,
        embedding_provider = provider,
    )

    modified_chunks = make_chunks()

    modified_chunks[0] = Chunk(
        chunk_id = 1,
        document_name = "paper.pdf",
        page_number = 1,
        text = "modified document text",
    )

    second_provider = make_provider()

    load_or_build_semantic_index(
        path = path,
        chunks = modified_chunks,
        embedding_provider = second_provider,
    )

    assert second_provider.calls == [
        [
            "modified document text",
            "vector retrieval",
        ]
    ]


def test_manager_rebuilds_index_when_embedding_model_changes(tmp_path):
    chunks = make_chunks()

    provider_a = make_provider(
        model_id = "model-a",
    )

    path = tmp_path / "semantic_index.npz"

    load_or_build_semantic_index(
        path = path,
        chunks = chunks,
        embedding_provider = provider_a,
    )

    provider_b = make_provider(
        model_id = "model-b",
    )

    load_or_build_semantic_index(
        path = path,
        chunks = chunks,
        embedding_provider = provider_b,
    )

    assert provider_b.calls == [
        [
            "semantic search",
            "vector retrieval",
        ]
    ]


def test_manager_overwrites_stale_index_with_new_valid_index(tmp_path):
    chunks = make_chunks()

    provider_a = make_provider(
        model_id = "model-a",
    )

    path = tmp_path / "semantic_index.npz"

    load_or_build_semantic_index(
        path = path,
        chunks = chunks,
        embedding_provider = provider_a,
    )

    provider_b = make_provider(
        model_id = "model-b",
    )

    load_or_build_semantic_index(
        path = path,
        chunks = chunks,
        embedding_provider = provider_b,
    )

    third_provider = make_provider(
        model_id = "model-b",
    )

    load_or_build_semantic_index(
        path = path,
        chunks = chunks,
        embedding_provider = third_provider,
    )

    assert third_provider.calls == []

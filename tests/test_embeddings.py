import pytest
from langchain_core.embeddings import Embeddings

from agentic_research_rag.config import Settings
from agentic_research_rag.retrieval import embeddings as embeddings_module


class FakeEmbeddings(Embeddings):
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [1.0, 0.0]
            for _ in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [1.0, 0.0]


def test_build_local_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_huggingface_embeddings(
        model_name: str,
    ) -> FakeEmbeddings:
        captured["model_name"] = model_name
        return FakeEmbeddings()

    monkeypatch.setattr(
        embeddings_module,
        "HuggingFaceEmbeddings",
        fake_huggingface_embeddings,
    )

    settings = Settings(
        embedding_backend = "local",
        local_embedding_model = "local-test-model",
    )

    embeddings = embeddings_module.build_embeddings(
        settings = settings
    )

    assert isinstance(embeddings, FakeEmbeddings)
    assert captured["model_name"] == "local-test-model"


def test_build_openai_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_openai_embeddings(
        model: str,
    ) -> FakeEmbeddings:
        captured["model"] = model
        return FakeEmbeddings()

    monkeypatch.setattr(
        embeddings_module,
        "OpenAIEmbeddings",
        fake_openai_embeddings,
    )

    settings = Settings(
        embedding_backend = "openai",
        openai_embedding_model = "openai-test-model",
    )

    embeddings = embeddings_module.build_embeddings(
        settings = settings
    )

    assert isinstance(embeddings, FakeEmbeddings)
    assert captured["model"] == "openai-test-model"


def test_fake_embeddings_follow_langchain_interface() -> None:
    embeddings: Embeddings = FakeEmbeddings()

    document_vectors = embeddings.embed_documents(
        ["first", "second"]
    )

    query_vector = embeddings.embed_query(
        "query"
    )

    assert document_vectors == [
        [1.0, 0.0],
        [1.0, 0.0],
    ]

    assert query_vector == [1.0, 0.0]
import pytest

from agentic_research_rag.config import Settings


def test_default_settings() -> None:
    settings = Settings()

    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 200

    assert settings.embedding_backend == "local"
    assert settings.local_embedding_model == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert settings.openai_embedding_model == "text-embedding-3-small"

    assert settings.cross_encoder_model == (
        "cross-encoder/ms-marco-MiniLM-L6-v2"
    )

    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_temperature == 0.0

    assert settings.rrf_k == 60
    assert settings.semantic_weight == 0.5
    assert settings.bm25_weight == 0.5

    assert settings.candidate_k == 30
    assert settings.rerank_k == 20
    assert settings.final_k == 5

    assert settings.memory_turns == 5


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")

    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "local-model")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "openai-model")

    monkeypatch.setenv("CROSS_ENCODER_MODEL", "reranker-model")

    monkeypatch.setenv("LLM_MODEL", "llm-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")

    monkeypatch.setenv("RRF_K", "50")
    monkeypatch.setenv("SEMANTIC_WEIGHT", "0.7")
    monkeypatch.setenv("BM25_WEIGHT", "0.3")

    monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", "40")
    monkeypatch.setenv("RERANK_K", "25")
    monkeypatch.setenv("FINAL_K", "8")

    monkeypatch.setenv("MEMORY_TURNS", "10")

    settings = Settings.from_env()

    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100

    assert settings.embedding_backend == "openai"
    assert settings.local_embedding_model == "local-model"
    assert settings.openai_embedding_model == "openai-model"

    assert settings.cross_encoder_model == "reranker-model"

    assert settings.llm_model == "llm-model"
    assert settings.llm_temperature == 0.2

    assert settings.rrf_k == 50
    assert settings.semantic_weight == 0.7
    assert settings.bm25_weight == 0.3

    assert settings.candidate_k == 40
    assert settings.rerank_k == 25
    assert settings.final_k == 8

    assert settings.memory_turns == 10


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_size", 0),
        ("chunk_size", -1),
        ("chunk_overlap", -1),
        ("rrf_k", 0),
        ("candidate_k", 0),
        ("rerank_k", 0),
        ("final_k", 0),
        ("memory_turns", -1),
    ],
)
def test_positive_numeric_validation(
    field: str,
    value: int,
) -> None:
    kwargs = {field: value}

    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        Settings(
            chunk_size = 100,
            chunk_overlap = 100,
        )


def test_embedding_backend_must_be_supported() -> None:
    with pytest.raises(ValueError):
        Settings(
            embedding_backend = "unknown",
        )


def test_retrieval_weights_cannot_both_be_zero() -> None:
    with pytest.raises(ValueError):
        Settings(
            semantic_weight = 0.0,
            bm25_weight = 0.0,
        )


def test_retrieval_weights_cannot_be_negative() -> None:
    with pytest.raises(ValueError):
        Settings(
            semantic_weight = -0.1,
        )


def test_rerank_k_cannot_exceed_combined_candidates() -> None:
    with pytest.raises(ValueError):
        Settings(
            candidate_k = 10,
            rerank_k = 21,
        )


def test_final_k_cannot_exceed_rerank_k() -> None:
    with pytest.raises(ValueError):
        Settings(
            rerank_k = 5,
            final_k = 6,
        )


@pytest.mark.parametrize(
    "temperature",
    [-0.1, 2.1],
)
def test_llm_temperature_range(temperature: float) -> None:
    with pytest.raises(ValueError):
        Settings(
            llm_temperature = temperature,
        )
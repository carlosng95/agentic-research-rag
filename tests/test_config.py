import pytest

from agentic_research_rag.config import Settings


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "1200")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    monkeypatch.setenv("RRF_K", "60")
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", "30")
    monkeypatch.setenv("RERANK_K", "20")
    monkeypatch.setenv("FINAL_K", "5")
    monkeypatch.setenv("MEMORY_TURNS", "5")

    settings = Settings.from_env()

    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 200
    assert settings.rrf_k == 60
    assert settings.candidate_k == 30
    assert settings.rerank_k == 20
    assert settings.final_k == 5
    assert settings.memory_turns == 5
    
def test_chunk_overlap_cannot_exceed_chunk_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")

    with pytest.raises(
        ValueError,
        match = "chunk_overlap must be smaller than chunk_size",
    ):
        Settings.from_env()
        
def test_rerank_k_cannot_exceed_candidate_k(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", "10")
    monkeypatch.setenv("RERANK_K", "20")

    with pytest.raises(
        ValueError,
        match = "rerank_k cannot be greater than candidate_k",
    ):
        Settings.from_env()
        
def test_final_k_cannot_exceed_rerank_k(monkeypatch):
    monkeypatch.setenv("RERANK_K", "5")
    monkeypatch.setenv("FINAL_K", "10")

    with pytest.raises(
        ValueError,
        match = "final_k cannot be greater than rerank_k",
    ):
        Settings.from_env()
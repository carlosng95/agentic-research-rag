from pathlib import Path

import pytest

from agentic_research_rag.artifacts.validate_candidate import (
    _expected_embedding_model_id,
    validate_candidate,
)
from agentic_research_rag.artifacts.validation import ArtifactValidationResult
from agentic_research_rag.config import Settings


def test_expected_embedding_model_id_for_local_backend():
    settings = Settings(
        embedding_backend = "local",
        local_embedding_model = "sentence-transformers/all-MiniLM-L6-v2",
    )

    result = _expected_embedding_model_id(settings = settings)

    assert result == "local:sentence-transformers/all-MiniLM-L6-v2"


def test_expected_embedding_model_id_for_openai_backend():
    settings = Settings(
        embedding_backend = "openai",
        openai_embedding_model = "text-embedding-3-small",
    )

    result = _expected_embedding_model_id(settings = settings)

    assert result == "openai:text-embedding-3-small"


def test_validate_candidate_downloads_and_validates(monkeypatch):
    settings = Settings(
        artifact_bucket = "test-bucket",
        artifact_prefix = "agentic-research-rag",
        embedding_backend = "local",
        local_embedding_model = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size = 1200,
        chunk_overlap = 200,
    )

    calls = {}

    def fake_download_index_artifact(version, destination_dir, settings, s3_client = None):
        calls["version"] = version
        calls["destination_dir"] = Path(destination_dir)
        calls["settings"] = settings
        calls["s3_client"] = s3_client

        assert calls["destination_dir"].is_dir()

    def fake_validate_index_artifact(
        index_dir,
        expected_embedding_model_id,
        expected_chunk_size,
        expected_chunk_overlap,
    ):
        calls["index_dir"] = Path(index_dir)
        calls["expected_embedding_model_id"] = expected_embedding_model_id
        calls["expected_chunk_size"] = expected_chunk_size
        calls["expected_chunk_overlap"] = expected_chunk_overlap

        assert calls["index_dir"].is_dir()

        return ArtifactValidationResult(
            document_count = 515,
            corpus_fingerprint = "a" * 64,
            embedding_model_id = expected_embedding_model_id,
        )

    monkeypatch.setattr(
        "agentic_research_rag.artifacts.validate_candidate.download_index_artifact",
        fake_download_index_artifact,
    )
    monkeypatch.setattr(
        "agentic_research_rag.artifacts.validate_candidate.validate_index_artifact",
        fake_validate_index_artifact,
    )

    result = validate_candidate(
        version = "20260916T035009Z-e8b5250b",
        settings = settings,
        s3_client = "fake-s3-client",
    )

    assert result.document_count == 515
    assert calls["version"] == "20260916T035009Z-e8b5250b"
    assert calls["settings"] is settings
    assert calls["s3_client"] == "fake-s3-client"
    assert calls["expected_embedding_model_id"] == "local:sentence-transformers/all-MiniLM-L6-v2"
    assert calls["expected_chunk_size"] == 1200
    assert calls["expected_chunk_overlap"] == 200


def test_validate_candidate_propagates_validation_failure(monkeypatch):
    settings = Settings(
        artifact_bucket = "test-bucket",
        embedding_backend = "local",
    )

    def fake_download_index_artifact(version, destination_dir, settings, s3_client = None):
        pass

    def fake_validate_index_artifact(**kwargs):
        raise ValueError("FAISS vector count mismatch")

    monkeypatch.setattr(
        "agentic_research_rag.artifacts.validate_candidate.download_index_artifact",
        fake_download_index_artifact,
    )
    monkeypatch.setattr(
        "agentic_research_rag.artifacts.validate_candidate.validate_index_artifact",
        fake_validate_index_artifact,
    )

    with pytest.raises(ValueError, match = "FAISS vector count mismatch"):
        validate_candidate(
            version = "candidate-version",
            settings = settings,
        )

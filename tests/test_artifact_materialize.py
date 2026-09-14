import json
from pathlib import Path

import pytest

from agentic_research_rag.artifacts import materialize
from agentic_research_rag.artifacts.materialize import materialize_index
from agentic_research_rag.config import Settings


def build_settings(artifact_source: str = "s3") -> Settings:
    return Settings(
        artifact_source = artifact_source,
        artifact_bucket = "test-bucket",
        artifact_prefix = "research",
    )


def build_manifest() -> dict:
    return {
        "index_version": "v001",
        "schema_version": 1,
        "corpus_fingerprint": "abc123",
        "embedding_model_id": "local:model",
        "document_count": 100,
    }


def write_artifact(index_dir: Path, document_count: int = 100) -> None:
    index_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    (index_dir / "index.faiss").write_bytes(b"faiss")
    (index_dir / "index.pkl").write_bytes(b"pickle")
    (index_dir / "documents.jsonl").write_text(
        '{"page_content": "test", "metadata": {}}\n',
        encoding = "utf-8",
    )

    metadata = {
        "schema_version": 1,
        "corpus_fingerprint": "abc123",
        "embedding_model_id": "local:model",
        "document_count": document_count,
    }

    (index_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding = "utf-8",
    )


def test_local_source_does_not_download(tmp_path, monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("S3 should not be called.")

    monkeypatch.setattr(
        materialize,
        "load_current_manifest",
        fail_if_called,
    )

    result = materialize_index(
        index_dir = tmp_path / "index",
        settings = build_settings(
            artifact_source = "local",
        ),
    )

    assert result is None


def test_s3_source_materializes_current_version(tmp_path, monkeypatch) -> None:
    index_dir = tmp_path / "index"

    monkeypatch.setattr(
        materialize,
        "load_current_manifest",
        lambda **kwargs: build_manifest(),
    )

    def fake_download(version, destination_dir, settings, s3_client = None):
        assert version == "v001"
        write_artifact(index_dir = Path(destination_dir))

    monkeypatch.setattr(
        materialize,
        "download_index_artifact",
        fake_download,
    )

    version = materialize_index(
        index_dir = index_dir,
        settings = build_settings(),
    )

    assert version == "v001"
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "index.pkl").exists()
    assert (index_dir / "documents.jsonl").exists()
    assert (index_dir / "metadata.json").exists()

    assert (
        index_dir / ".artifact-version"
    ).read_text(
        encoding = "utf-8"
    ) == "v001"


def test_matching_local_version_skips_download(tmp_path, monkeypatch) -> None:
    index_dir = tmp_path / "index"

    write_artifact(index_dir = index_dir)

    (index_dir / ".artifact-version").write_text(
        "v001",
        encoding = "utf-8",
    )

    monkeypatch.setattr(
        materialize,
        "load_current_manifest",
        lambda **kwargs: build_manifest(),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Artifact should not be downloaded again."
        )

    monkeypatch.setattr(
        materialize,
        "download_index_artifact",
        fail_if_called,
    )

    version = materialize_index(
        index_dir = index_dir,
        settings = build_settings(),
    )

    assert version == "v001"


def test_invalid_download_does_not_replace_existing_index(
    tmp_path,
    monkeypatch,
) -> None:
    index_dir = tmp_path / "index"

    index_dir.mkdir()
    (index_dir / "old-index.txt").write_text(
        "keep-me",
        encoding = "utf-8",
    )

    monkeypatch.setattr(
        materialize,
        "load_current_manifest",
        lambda **kwargs: build_manifest(),
    )

    def fake_download(version, destination_dir, settings, s3_client = None):
        write_artifact(
            index_dir = Path(destination_dir),
            document_count = 999,
        )

    monkeypatch.setattr(
        materialize,
        "download_index_artifact",
        fake_download,
    )

    with pytest.raises(
        ValueError,
        match = "document_count",
    ):
        materialize_index(
            index_dir = index_dir,
            settings = build_settings(),
        )

    assert (
        index_dir / "old-index.txt"
    ).read_text(
        encoding = "utf-8"
    ) == "keep-me"


def test_invalid_artifact_source_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match = "artifact_source",
    ):
        Settings(
            artifact_source = "invalid",
        )
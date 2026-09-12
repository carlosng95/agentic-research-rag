from datetime import datetime, timezone
import json

import pytest

from agentic_research_rag.artifacts.publish import (
    build_artifact_version,
    load_artifact_metadata,
)


def test_load_artifact_metadata(tmp_path) -> None:
    metadata = {
        "schema_version": 1,
        "corpus_fingerprint": "abcdef1234567890",
        "embedding_model_id": "local:model",
        "document_count": 100,
    }

    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding = "utf-8",
    )

    loaded = load_artifact_metadata(index_dir = tmp_path)

    assert loaded == metadata


def test_load_artifact_metadata_rejects_missing_fields(tmp_path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding = "utf-8",
    )

    with pytest.raises(ValueError, match = "corpus_fingerprint"):
        load_artifact_metadata(index_dir = tmp_path)


def test_build_artifact_version() -> None:
    metadata = {
        "corpus_fingerprint": "abcdef1234567890",
    }

    now = datetime(
        2026,
        9,
        12,
        4,
        30,
        0,
        tzinfo = timezone.utc,
    )

    version = build_artifact_version(
        metadata = metadata,
        now = now,
    )

    assert version == "20260912T043000Z-abcdef12"


def test_build_artifact_version_rejects_empty_fingerprint() -> None:
    metadata = {
        "corpus_fingerprint": "   ",
    }

    with pytest.raises(ValueError, match = "corpus_fingerprint"):
        build_artifact_version(metadata = metadata)
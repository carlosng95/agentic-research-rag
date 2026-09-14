import io
import json
from pathlib import Path

import pytest

from agentic_research_rag.artifacts.s3_store import (
    download_index_artifact,
    load_current_manifest,
    promote_index,
    upload_index_artifact,
)
from agentic_research_rag.config import Settings


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        self.objects[(Bucket, Key)] = Path(Filename).read_bytes()

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        data = self.objects[(Bucket, Key)]

        destination = Path(Filename)
        destination.parent.mkdir(
            parents = True,
            exist_ok = True,
        )
        destination.write_bytes(data)

    def head_object(self, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self.objects:
            raise FileNotFoundError(
                f"S3 object not found: s3://{Bucket}/{Key}"
            )

        return {}

    def put_object(
        self,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
    ) -> None:
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket: str, Key: str) -> dict:
        data = self.objects[(Bucket, Key)]

        return {
            "Body": io.BytesIO(data),
        }


def build_settings() -> Settings:
    return Settings(
        aws_region = "us-east-1",
        artifact_bucket = "test-bucket",
        artifact_prefix = "research",
    )


def create_artifact(index_dir: Path) -> None:
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
    (index_dir / "metadata.json").write_text(
        "{}",
        encoding = "utf-8",
    )


def test_upload_index_artifact_uploads_all_files(tmp_path) -> None:
    index_dir = tmp_path / "index"
    create_artifact(index_dir = index_dir)

    client = FakeS3Client()
    settings = build_settings()

    upload_index_artifact(
        index_dir = index_dir,
        version = "v001",
        settings = settings,
        s3_client = client,
    )

    expected_keys = {
        (
            "test-bucket",
            "research/indexes/versions/v001/index.faiss",
        ),
        (
            "test-bucket",
            "research/indexes/versions/v001/index.pkl",
        ),
        (
            "test-bucket",
            "research/indexes/versions/v001/documents.jsonl",
        ),
        (
            "test-bucket",
            "research/indexes/versions/v001/metadata.json",
        ),
    }

    assert set(client.objects) == expected_keys


def test_upload_rejects_incomplete_artifact_before_uploading(tmp_path) -> None:
    index_dir = tmp_path / "index"
    create_artifact(index_dir = index_dir)

    (index_dir / "metadata.json").unlink()

    client = FakeS3Client()

    with pytest.raises(FileNotFoundError, match = "metadata.json"):
        upload_index_artifact(
            index_dir = index_dir,
            version = "v001",
            settings = build_settings(),
            s3_client = client,
        )

    assert client.objects == {}


def test_download_index_artifact_downloads_all_files(tmp_path) -> None:
    source_dir = tmp_path / "source"
    destination_dir = tmp_path / "destination"

    create_artifact(index_dir = source_dir)

    client = FakeS3Client()
    settings = build_settings()

    upload_index_artifact(
        index_dir = source_dir,
        version = "v001",
        settings = settings,
        s3_client = client,
    )

    download_index_artifact(
        version = "v001",
        destination_dir = destination_dir,
        settings = settings,
        s3_client = client,
    )

    assert (destination_dir / "index.faiss").read_bytes() == b"faiss"
    assert (destination_dir / "index.pkl").read_bytes() == b"pickle"
    assert (destination_dir / "documents.jsonl").exists()
    assert (destination_dir / "metadata.json").exists()


def test_promote_and_load_current_manifest(tmp_path) -> None:
    index_dir = tmp_path / "index"
    create_artifact(index_dir = index_dir)

    client = FakeS3Client()
    settings = build_settings()

    upload_index_artifact(
        index_dir = index_dir,
        version = "v001",
        settings = settings,
        s3_client = client,
    )

    metadata = {
        "schema_version": 1,
        "corpus_fingerprint": "abc123",
        "embedding_model_id": "local:model",
        "document_count": 100,
    }

    promote_index(
        version = "v001",
        settings = settings,
        metadata = metadata,
        s3_client = client,
    )

    manifest = load_current_manifest(
        settings = settings,
        s3_client = client,
    )

    assert manifest == {
        "index_version": "v001",
        "corpus_fingerprint": "abc123",
        "embedding_model_id": "local:model",
        "document_count": 100,
        "schema_version": 1,
    }


def test_promote_rejects_missing_remote_artifact() -> None:
    client = FakeS3Client()

    metadata = {
        "schema_version": 1,
        "corpus_fingerprint": "abc123",
        "embedding_model_id": "local:model",
        "document_count": 100,
    }

    with pytest.raises(FileNotFoundError):
        promote_index(
            version = "v001",
            settings = build_settings(),
            metadata = metadata,
            s3_client = client,
        )


def test_s3_operations_require_artifact_bucket(tmp_path) -> None:
    index_dir = tmp_path / "index"
    create_artifact(index_dir = index_dir)

    settings = Settings(
        artifact_bucket = "",
    )

    with pytest.raises(ValueError, match = "artifact_bucket"):
        upload_index_artifact(
            index_dir = index_dir,
            version = "v001",
            settings = settings,
            s3_client = FakeS3Client(),
        )


def test_s3_operations_reject_empty_version(tmp_path) -> None:
    index_dir = tmp_path / "index"
    create_artifact(index_dir = index_dir)

    with pytest.raises(ValueError, match = "version"):
        upload_index_artifact(
            index_dir = index_dir,
            version = "   ",
            settings = build_settings(),
            s3_client = FakeS3Client(),
        )
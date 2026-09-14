import json
from pathlib import Path
from typing import Any

import boto3

from ..config import Settings


_ARTIFACT_FILES = (
    "index.faiss",
    "index.pkl",
    "documents.jsonl",
    "metadata.json",
)


def build_s3_client(settings: Settings) -> Any:
    return boto3.client(
        "s3",
        region_name = settings.aws_region,
    )


def _require_bucket(settings: Settings) -> None:
    if not settings.artifact_bucket.strip():
        raise ValueError("artifact_bucket must be configured.")


def _require_version(version: str) -> str:
    version = version.strip()

    if not version:
        raise ValueError("version cannot be empty.")

    return version


def _artifact_prefix(settings: Settings, version: str) -> str:
    base_prefix = settings.artifact_prefix.strip("/")

    return f"{base_prefix}/indexes/versions/{version}"


def _manifest_key(settings: Settings) -> str:
    base_prefix = settings.artifact_prefix.strip("/")

    return f"{base_prefix}/indexes/manifests/current.json"


def _validate_local_artifact(index_dir: Path) -> dict[str, Path]:
    paths = {
        filename: index_dir / filename
        for filename in _ARTIFACT_FILES
    }

    missing = [
        str(path)
        for path in paths.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing artifact files: "
            + ", ".join(missing)
        )

    return paths


def _validate_remote_artifact(version: str, settings: Settings, s3_client: Any) -> None:
    prefix = _artifact_prefix(
        settings = settings,
        version = version,
    )

    for filename in _ARTIFACT_FILES:
        s3_client.head_object(
            Bucket = settings.artifact_bucket,
            Key = f"{prefix}/{filename}",
        )


def upload_index_artifact(
    index_dir: str | Path,
    version: str,
    settings: Settings,
    s3_client: Any | None = None,
) -> None:
    _require_bucket(settings = settings)
    version = _require_version(version = version)

    index_dir = Path(index_dir)

    paths = _validate_local_artifact(
        index_dir = index_dir,
    )

    if s3_client is None:
        s3_client = build_s3_client(
            settings = settings,
        )

    prefix = _artifact_prefix(
        settings = settings,
        version = version,
    )

    for filename, path in paths.items():
        s3_client.upload_file(
            Filename = str(path),
            Bucket = settings.artifact_bucket,
            Key = f"{prefix}/{filename}",
        )


def download_index_artifact(
    version: str,
    destination_dir: str | Path,
    settings: Settings,
    s3_client: Any | None = None,
) -> None:
    _require_bucket(settings = settings)
    version = _require_version(version = version)

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    if s3_client is None:
        s3_client = build_s3_client(
            settings = settings,
        )

    prefix = _artifact_prefix(
        settings = settings,
        version = version,
    )

    for filename in _ARTIFACT_FILES:
        s3_client.download_file(
            Bucket = settings.artifact_bucket,
            Key = f"{prefix}/{filename}",
            Filename = str(destination_dir / filename),
        )


def promote_index(
    version: str,
    settings: Settings,
    metadata: dict[str, Any],
    s3_client: Any | None = None,
) -> None:
    _require_bucket(settings = settings)
    version = _require_version(version = version)

    if s3_client is None:
        s3_client = build_s3_client(
            settings = settings,
        )

    _validate_remote_artifact(
        version = version,
        settings = settings,
        s3_client = s3_client,
    )

    manifest = {
        "index_version": version,
        "corpus_fingerprint": metadata["corpus_fingerprint"],
        "embedding_model_id": metadata["embedding_model_id"],
        "document_count": metadata["document_count"],
        "schema_version": metadata["schema_version"],
    }

    s3_client.put_object(
        Bucket = settings.artifact_bucket,
        Key = _manifest_key(settings = settings),
        Body = json.dumps(
            manifest,
            indent = 2,
            sort_keys = True,
        ).encode("utf-8"),
        ContentType = "application/json",
    )


def load_current_manifest(
    settings: Settings,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    _require_bucket(settings = settings)

    if s3_client is None:
        s3_client = build_s3_client(
            settings = settings,
        )

    response = s3_client.get_object(
        Bucket = settings.artifact_bucket,
        Key = _manifest_key(settings = settings),
    )

    return json.loads(
        response["Body"].read().decode("utf-8")
    )
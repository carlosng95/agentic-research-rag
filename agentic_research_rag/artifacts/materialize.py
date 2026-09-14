import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from ..config import Settings
from .s3_store import download_index_artifact, load_current_manifest


_ARTIFACT_FILES = (
    "index.faiss",
    "index.pkl",
    "documents.jsonl",
    "metadata.json",
)

_VERSION_MARKER = ".artifact-version"

_MANIFEST_METADATA_FIELDS = (
    "schema_version",
    "corpus_fingerprint",
    "embedding_model_id",
    "document_count",
)


def _artifact_is_complete(index_dir: Path) -> bool:
    return all(
        (index_dir / filename).is_file()
        for filename in _ARTIFACT_FILES
    )


def _read_local_version(index_dir: Path) -> str | None:
    marker = index_dir / _VERSION_MARKER

    if not marker.is_file():
        return None

    version = marker.read_text(encoding = "utf-8").strip()

    return version or None


def _load_metadata(index_dir: Path) -> dict[str, Any]:
    metadata_path = index_dir / "metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    return json.loads(
        metadata_path.read_text(encoding = "utf-8")
    )


def _validate_manifest_against_metadata(
    manifest: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    for field in _MANIFEST_METADATA_FIELDS:
        if field not in manifest:
            raise ValueError(
                f"Current manifest is missing required field: {field}"
            )

        if metadata.get(field) != manifest[field]:
            raise ValueError(
                f"Downloaded artifact metadata does not match "
                f"the current manifest for field: {field}"
            )


def materialize_index(
    index_dir: str | Path,
    settings: Settings,
    s3_client: Any | None = None,
) -> str | None:
    index_dir = Path(index_dir)

    if settings.artifact_source == "local":
        return None

    manifest = load_current_manifest(
        settings = settings,
        s3_client = s3_client,
    )

    version = str(
        manifest.get("index_version", "")
    ).strip()

    if not version:
        raise ValueError(
            "Current manifest does not contain a valid index_version."
        )

    if (
        _read_local_version(index_dir = index_dir) == version
        and _artifact_is_complete(index_dir = index_dir)
    ):
        metadata = _load_metadata(
            index_dir = index_dir,
        )

        _validate_manifest_against_metadata(
            manifest = manifest,
            metadata = metadata,
        )

        return version

    index_dir.parent.mkdir(
        parents = True,
        exist_ok = True,
    )

    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix = f".{index_dir.name}-download-",
            dir = index_dir.parent,
        )
    )

    try:
        download_index_artifact(
            version = version,
            destination_dir = temporary_dir,
            settings = settings,
            s3_client = s3_client,
        )

        if not _artifact_is_complete(
            index_dir = temporary_dir,
        ):
            raise RuntimeError(
                "Downloaded index artifact is incomplete."
            )

        metadata = _load_metadata(
            index_dir = temporary_dir,
        )

        _validate_manifest_against_metadata(
            manifest = manifest,
            metadata = metadata,
        )

        (temporary_dir / _VERSION_MARKER).write_text(
            version,
            encoding = "utf-8",
        )

        if index_dir.exists():
            shutil.rmtree(index_dir)

        shutil.move(
            str(temporary_dir),
            str(index_dir),
        )

    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    return version
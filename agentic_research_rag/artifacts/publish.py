import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ..config import Settings
from .s3_store import promote_index, upload_index_artifact


_REQUIRED_METADATA_FIELDS = (
    "schema_version",
    "corpus_fingerprint",
    "embedding_model_id",
    "document_count",
)


def load_artifact_metadata(index_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(index_dir) / "metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding = "utf-8"))

    missing = [field for field in _REQUIRED_METADATA_FIELDS if field not in metadata]

    if missing:
        raise ValueError(
            "Artifact metadata is missing required fields: "
            + ", ".join(missing)
        )

    return metadata


def build_artifact_version(metadata: dict[str, Any], now: datetime | None = None) -> str:
    fingerprint = str(metadata["corpus_fingerprint"]).strip()

    if not fingerprint:
        raise ValueError("corpus_fingerprint cannot be empty.")

    if now is None:
        now = datetime.now(timezone.utc)

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware.")

    timestamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    return f"{timestamp}-{fingerprint[:8]}"


def upload_artifact(
    index_dir: str | Path,
    version: str | None = None,
    settings: Settings | None = None,
) -> str:
    if settings is None:
        settings = Settings.from_env()

    metadata = load_artifact_metadata(index_dir = index_dir)

    if version is None:
        version = build_artifact_version(metadata = metadata)

    upload_index_artifact(
        index_dir = index_dir,
        version = version,
        settings = settings,
    )

    return version


def promote_artifact(
    index_dir: str | Path,
    version: str,
    settings: Settings | None = None,
) -> None:
    if settings is None:
        settings = Settings.from_env()

    metadata = load_artifact_metadata(index_dir = index_dir)

    promote_index(
        version = version,
        settings = settings,
        metadata = metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Publish versioned retrieval artifacts to S3.")

    subparsers = parser.add_subparsers(dest = "command", required = True)

    upload_parser = subparsers.add_parser(
        "upload",
        help = "Upload a local index artifact as a new version.",
    )
    upload_parser.add_argument(
        "--index-dir",
        default = "data/indexes/faiss",
        help = "Directory containing the local index artifact.",
    )
    upload_parser.add_argument(
        "--version",
        default = None,
        help = "Optional explicit version. Generated automatically when omitted.",
    )

    promote_parser = subparsers.add_parser(
        "promote",
        help = "Promote an uploaded version as the current index.",
    )
    promote_parser.add_argument(
        "--index-dir",
        default = "data/indexes/faiss",
        help = "Directory containing metadata.json for the promoted artifact.",
    )
    promote_parser.add_argument(
        "--version",
        required = True,
        help = "Uploaded version to promote.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "upload":
        version = upload_artifact(
            index_dir = args.index_dir,
            version = args.version,
        )

        print("Index artifact uploaded successfully.")
        print(f"Version: {version}")
        return

    promote_artifact(
        index_dir = args.index_dir,
        version = args.version,
    )

    print("Index artifact promoted successfully.")
    print(f"Version: {args.version}")


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..config import Settings
from .s3_store import download_index_artifact
from .validation import ArtifactValidationResult, validate_index_artifact


def _expected_embedding_model_id(settings: Settings) -> str:
    if settings.embedding_backend == "local":
        return f"local:{settings.local_embedding_model}"

    if settings.embedding_backend == "openai":
        return f"openai:{settings.openai_embedding_model}"

    raise ValueError(f"Unsupported embedding backend: {settings.embedding_backend}")


def validate_candidate(
    version: str,
    settings: Settings | None = None,
    s3_client: Any | None = None,
) -> ArtifactValidationResult:
    if settings is None:
        settings = Settings.from_env()

    with TemporaryDirectory(prefix = "agentic-rag-candidate-") as temp_dir:
        index_dir = Path(temp_dir)

        download_index_artifact(
            version = version,
            destination_dir = index_dir,
            settings = settings,
            s3_client = s3_client,
        )

        return validate_index_artifact(
            index_dir = index_dir,
            expected_embedding_model_id = _expected_embedding_model_id(settings = settings),
            expected_chunk_size = settings.chunk_size,
            expected_chunk_overlap = settings.chunk_overlap,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Validate a versioned retrieval index candidate from S3.")
    parser.add_argument("--version", required = True, help = "Versioned index artifact to validate.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = validate_candidate(version = args.version)

    print("Candidate index validation succeeded.")
    print(f"Version: {args.version}")
    print(f"Documents: {result.document_count}")
    print(f"Fingerprint: {result.corpus_fingerprint}")
    print(f"Embedding model: {result.embedding_model_id}")


if __name__ == "__main__":
    main()

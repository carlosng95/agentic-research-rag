from dataclasses import dataclass
import json
from pathlib import Path
import re

import faiss


REQUIRED_FILES = (
    "documents.jsonl",
    "index.faiss",
    "index.pkl",
    "metadata.json",
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ArtifactValidationResult:
    document_count: int
    corpus_fingerprint: str
    embedding_model_id: str


def _require_file(index_dir: Path, filename: str) -> Path:
    path = index_dir / filename

    if not path.is_file():
        raise ValueError(f"Required artifact file is missing: {filename}")

    if path.stat().st_size == 0:
        raise ValueError(f"Artifact file is empty: {filename}")

    return path


def _load_metadata(metadata_path: Path) -> dict:
    try:
        metadata = json.loads(metadata_path.read_text(encoding = "utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("metadata.json is not valid JSON.") from exc

    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object.")

    return metadata


def _require_metadata_field(metadata: dict, field: str):
    if field not in metadata:
        raise ValueError(f"metadata.json is missing required field: {field}")

    return metadata[field]


def validate_index_artifact(
    index_dir: Path,
    expected_embedding_model_id: str,
    expected_chunk_size: int,
    expected_chunk_overlap: int,
) -> ArtifactValidationResult:
    index_dir = Path(index_dir)

    if not index_dir.is_dir():
        raise ValueError(f"Index directory does not exist: {index_dir}")

    artifact_paths = {
        filename: _require_file(index_dir, filename)
        for filename in REQUIRED_FILES
    }

    metadata = _load_metadata(artifact_paths["metadata.json"])

    schema_version = _require_metadata_field(metadata, "schema_version")
    document_count = _require_metadata_field(metadata, "document_count")
    corpus_fingerprint = _require_metadata_field(metadata, "corpus_fingerprint")
    embedding_model_id = _require_metadata_field(metadata, "embedding_model_id")
    chunk_size = _require_metadata_field(metadata, "chunk_size")
    chunk_overlap = _require_metadata_field(metadata, "chunk_overlap")

    if schema_version != 1:
        raise ValueError(f"Unsupported artifact schema version: {schema_version}")

    if not isinstance(document_count, int) or document_count <= 0:
        raise ValueError(f"Invalid document count: {document_count}")

    if not isinstance(corpus_fingerprint, str) or not SHA256_PATTERN.fullmatch(corpus_fingerprint):
        raise ValueError("Corpus fingerprint must be a lowercase SHA-256 hex digest.")

    if embedding_model_id != expected_embedding_model_id:
        raise ValueError(
            f"Embedding model mismatch: expected {expected_embedding_model_id}, "
            f"got {embedding_model_id}"
        )

    if chunk_size != expected_chunk_size:
        raise ValueError(f"Chunk size mismatch: expected {expected_chunk_size}, got {chunk_size}")

    if chunk_overlap != expected_chunk_overlap:
        raise ValueError(
            f"Chunk overlap mismatch: expected {expected_chunk_overlap}, got {chunk_overlap}"
        )

    with artifact_paths["documents.jsonl"].open("r", encoding = "utf-8") as file:
        jsonl_document_count = sum(1 for line in file if line.strip())

    if jsonl_document_count != document_count:
        raise ValueError(
            f"Document count mismatch: metadata={document_count}, "
            f"documents.jsonl={jsonl_document_count}"
        )

    faiss_index = faiss.read_index(str(artifact_paths["index.faiss"]))

    if faiss_index.ntotal != document_count:
        raise ValueError(
            f"FAISS vector count mismatch: metadata={document_count}, "
            f"faiss={faiss_index.ntotal}"
        )

    return ArtifactValidationResult(
        document_count = document_count,
        corpus_fingerprint = corpus_fingerprint,
        embedding_model_id = embedding_model_id,
    )

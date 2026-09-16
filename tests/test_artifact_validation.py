import json

import faiss
import numpy as np
import pytest

from agentic_research_rag.artifacts.validation import validate_index_artifact


FINGERPRINT = "a" * 64
EMBEDDING_MODEL_ID = "local:sentence-transformers/all-MiniLM-L6-v2"


def build_valid_artifact(tmp_path, document_count: int = 3):
    documents = [
        {"id": index, "text": f"Document {index}"}
        for index in range(document_count)
    ]

    with (tmp_path / "documents.jsonl").open("w", encoding = "utf-8") as file:
        for document in documents:
            file.write(json.dumps(document) + "\n")

    dimension = 4
    vectors = np.arange(document_count * dimension, dtype = np.float32).reshape(document_count, dimension)

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    faiss.write_index(index, str(tmp_path / "index.faiss"))

    (tmp_path / "index.pkl").write_bytes(b"test-index-metadata")

    metadata = {
        "schema_version": 1,
        "document_count": document_count,
        "corpus_fingerprint": FINGERPRINT,
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "chunk_size": 1200,
        "chunk_overlap": 200,
    }

    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding = "utf-8",
    )


def validate(tmp_path):
    return validate_index_artifact(
        index_dir = tmp_path,
        expected_embedding_model_id = EMBEDDING_MODEL_ID,
        expected_chunk_size = 1200,
        expected_chunk_overlap = 200,
    )


def test_validate_index_artifact_accepts_valid_artifact(tmp_path):
    build_valid_artifact(tmp_path)

    result = validate(tmp_path)

    assert result.document_count == 3
    assert result.corpus_fingerprint == FINGERPRINT
    assert result.embedding_model_id == EMBEDDING_MODEL_ID


def test_validate_index_artifact_rejects_missing_file(tmp_path):
    build_valid_artifact(tmp_path)
    (tmp_path / "index.pkl").unlink()

    with pytest.raises(ValueError, match = "Required artifact file is missing: index.pkl"):
        validate(tmp_path)


def test_validate_index_artifact_rejects_document_count_mismatch(tmp_path):
    build_valid_artifact(tmp_path)

    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding = "utf-8"))
    metadata["document_count"] = 4
    metadata_path.write_text(json.dumps(metadata), encoding = "utf-8")

    with pytest.raises(ValueError, match = "Document count mismatch"):
        validate(tmp_path)


def test_validate_index_artifact_rejects_faiss_count_mismatch(tmp_path):
    build_valid_artifact(tmp_path)

    index = faiss.IndexFlatIP(4)
    index.add(np.ones((2, 4), dtype = np.float32))
    faiss.write_index(index, str(tmp_path / "index.faiss"))

    with pytest.raises(ValueError, match = "FAISS vector count mismatch"):
        validate(tmp_path)


def test_validate_index_artifact_rejects_embedding_model_mismatch(tmp_path):
    build_valid_artifact(tmp_path)

    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding = "utf-8"))
    metadata["embedding_model_id"] = "unexpected-model"
    metadata_path.write_text(json.dumps(metadata), encoding = "utf-8")

    with pytest.raises(ValueError, match = "Embedding model mismatch"):
        validate(tmp_path)


def test_validate_index_artifact_rejects_invalid_fingerprint(tmp_path):
    build_valid_artifact(tmp_path)

    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding = "utf-8"))
    metadata["corpus_fingerprint"] = "invalid"
    metadata_path.write_text(json.dumps(metadata), encoding = "utf-8")

    with pytest.raises(ValueError, match = "Corpus fingerprint"):
        validate(tmp_path)

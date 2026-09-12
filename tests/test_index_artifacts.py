import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from agentic_research_rag.config import Settings
from agentic_research_rag.retrieval import index_manager


class FakeVectorStore:
    def save_local(self, folder_path: str, index_name: str) -> None:
        index_dir = Path(folder_path)

        (index_dir / f"{index_name}.faiss").write_bytes(b"fake-faiss")
        (index_dir / f"{index_name}.pkl").write_bytes(b"fake-pickle")


def build_test_artifact(index_dir: Path, settings: Settings) -> list[Document]:
    documents = [
        Document(
            page_content = "First research chunk.",
            metadata = {
                "chunk_id": 0,
                "source": "research.pdf",
                "page_number": 1,
            },
        ),
        Document(
            page_content = "Second research chunk.",
            metadata = {
                "chunk_id": 1,
                "source": "research.pdf",
                "page_number": 2,
            },
        ),
    ]

    index_manager.save_vector_store(
        vector_store = FakeVectorStore(),
        index_dir = index_dir,
        documents = documents,
        settings = settings,
    )

    return documents


def test_load_index_artifacts_returns_vector_store_and_documents(monkeypatch, tmp_path: Path) -> None:
    settings = Settings()
    index_dir = tmp_path / "index"

    documents = build_test_artifact(
        index_dir = index_dir,
        settings = settings,
    )

    embeddings = object()
    vector_store = object()

    monkeypatch.setattr(
        index_manager,
        "load_vector_store",
        lambda **kwargs: vector_store,
    )

    loaded_vector_store, loaded_documents = index_manager.load_index_artifacts(
        index_dir = index_dir,
        embeddings = embeddings,
        settings = settings,
    )

    assert loaded_vector_store is vector_store
    assert loaded_documents == documents


def test_load_index_artifacts_rejects_missing_file(monkeypatch, tmp_path: Path) -> None:
    settings = Settings()
    index_dir = tmp_path / "index"

    build_test_artifact(
        index_dir = index_dir,
        settings = settings,
    )

    (index_dir / "documents.jsonl").unlink()

    monkeypatch.setattr(
        index_manager,
        "load_vector_store",
        lambda **kwargs: object(),
    )

    with pytest.raises(FileNotFoundError, match = "documents.jsonl"):
        index_manager.load_index_artifacts(
            index_dir = index_dir,
            embeddings = object(),
            settings = settings,
        )


def test_load_index_artifacts_rejects_embedding_model_mismatch(monkeypatch, tmp_path: Path) -> None:
    build_settings = Settings(
        local_embedding_model = "sentence-transformers/all-MiniLM-L6-v2",
    )

    serving_settings = Settings(
        local_embedding_model = "sentence-transformers/another-model",
    )

    index_dir = tmp_path / "index"

    build_test_artifact(
        index_dir = index_dir,
        settings = build_settings,
    )

    monkeypatch.setattr(
        index_manager,
        "load_vector_store",
        lambda **kwargs: object(),
    )

    with pytest.raises(ValueError, match = "Embedding model mismatch"):
        index_manager.load_index_artifacts(
            index_dir = index_dir,
            embeddings = object(),
            settings = serving_settings,
        )


def test_load_index_artifacts_rejects_document_count_mismatch(monkeypatch, tmp_path: Path) -> None:
    settings = Settings()
    index_dir = tmp_path / "index"

    build_test_artifact(
        index_dir = index_dir,
        settings = settings,
    )

    metadata_path = index_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding = "utf-8"))

    metadata["document_count"] = 999

    metadata_path.write_text(
        json.dumps(metadata, indent = 2, sort_keys = True),
        encoding = "utf-8",
    )

    monkeypatch.setattr(
        index_manager,
        "load_vector_store",
        lambda **kwargs: object(),
    )

    with pytest.raises(ValueError, match = "document count mismatch"):
        index_manager.load_index_artifacts(
            index_dir = index_dir,
            embeddings = object(),
            settings = settings,
        )


def test_load_index_artifacts_rejects_corrupted_documents(monkeypatch, tmp_path: Path) -> None:
    settings = Settings()
    index_dir = tmp_path / "index"

    build_test_artifact(
        index_dir = index_dir,
        settings = settings,
    )

    documents_path = index_dir / "documents.jsonl"
    lines = documents_path.read_text(encoding = "utf-8").splitlines()

    payload = json.loads(lines[0])
    payload["page_content"] = "Tampered content."

    lines[0] = json.dumps(payload, ensure_ascii = False)

    documents_path.write_text(
        "\n".join(lines) + "\n",
        encoding = "utf-8",
    )

    monkeypatch.setattr(
        index_manager,
        "load_vector_store",
        lambda **kwargs: object(),
    )

    with pytest.raises(ValueError, match = "fingerprint"):
        index_manager.load_index_artifacts(
            index_dir = index_dir,
            embeddings = object(),
            settings = settings,
        )


def test_load_index_artifacts_rejects_unsupported_schema_version(monkeypatch, tmp_path: Path) -> None:
    settings = Settings()
    index_dir = tmp_path / "index"

    build_test_artifact(
        index_dir = index_dir,
        settings = settings,
    )

    metadata_path = index_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding = "utf-8"))

    metadata["schema_version"] = 999

    metadata_path.write_text(
        json.dumps(metadata, indent = 2, sort_keys = True),
        encoding = "utf-8",
    )

    monkeypatch.setattr(
        index_manager,
        "load_vector_store",
        lambda **kwargs: object(),
    )

    with pytest.raises(ValueError, match = "Unsupported index schema version"):
        index_manager.load_index_artifacts(
            index_dir = index_dir,
            embeddings = object(),
            settings = settings,
        )
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from agentic_research_rag.config import Settings
from agentic_research_rag.indexing import build as index_build


def test_build_index_wires_components(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        chunk_size = 500,
        chunk_overlap = 50,
    )

    papers_dir = tmp_path / "papers"
    index_dir = tmp_path / "indexes"

    documents = [
        Document(
            page_content = "Research evidence.",
            metadata = {
                "chunk_id": 0,
                "source": "research.pdf",
                "page_number": 1,
            },
        )
    ]

    embeddings = object()
    vector_store = object()

    calls: dict[str, Any] = {}

    def fake_load_corpus(papers_dir: Path, chunk_size: int, chunk_overlap: int):
        calls["load_corpus"] = {
            "papers_dir": papers_dir,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

        return documents

    def fake_build_embeddings(settings: Settings):
        calls["build_embeddings"] = {
            "settings": settings,
        }

        return embeddings

    def fake_build_vector_store(documents, embeddings):
        calls["build_vector_store"] = {
            "documents": documents,
            "embeddings": embeddings,
        }

        return vector_store

    def fake_save_vector_store(vector_store, index_dir, documents, settings):
        calls["save_vector_store"] = {
            "vector_store": vector_store,
            "index_dir": index_dir,
            "documents": documents,
            "settings": settings,
        }

    monkeypatch.setattr(index_build, "load_corpus", fake_load_corpus)
    monkeypatch.setattr(index_build, "build_embeddings", fake_build_embeddings)
    monkeypatch.setattr(index_build, "build_vector_store", fake_build_vector_store)
    monkeypatch.setattr(index_build, "save_vector_store", fake_save_vector_store)

    result = index_build.build_index(
        papers_dir = papers_dir,
        index_dir = index_dir,
        settings = settings,
    )

    assert result is None

    assert calls["load_corpus"] == {
        "papers_dir": papers_dir,
        "chunk_size": 500,
        "chunk_overlap": 50,
    }

    assert calls["build_embeddings"] == {
        "settings": settings,
    }

    assert calls["build_vector_store"] == {
        "documents": documents,
        "embeddings": embeddings,
    }

    assert calls["save_vector_store"] == {
        "vector_store": vector_store,
        "index_dir": index_dir,
        "documents": documents,
        "settings": settings,
    }


def test_build_index_refuses_to_replace_existing_index_without_force(monkeypatch, tmp_path: Path) -> None:
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()

    existing_file = index_dir / "existing.txt"
    existing_file.write_text("existing artifact", encoding = "utf-8")

    def fail_if_called(**kwargs):
        raise AssertionError("Indexing should not start.")

    monkeypatch.setattr(index_build, "load_corpus", fail_if_called)

    with pytest.raises(FileExistsError, match = "Use --force"):
        index_build.build_index(
            papers_dir = tmp_path / "papers",
            index_dir = index_dir,
            settings = Settings(),
        )

    assert existing_file.exists()


def test_build_index_replaces_existing_index_with_force(monkeypatch, tmp_path: Path) -> None:
    settings = Settings()

    papers_dir = tmp_path / "papers"
    index_dir = tmp_path / "indexes"

    index_dir.mkdir()

    stale_file = index_dir / "stale.txt"
    stale_file.write_text("stale", encoding = "utf-8")

    documents = [
        Document(
            page_content = "Updated evidence.",
            metadata = {
                "chunk_id": 0,
                "source": "updated.pdf",
                "page_number": 1,
            },
        )
    ]

    embeddings = object()
    vector_store = object()

    monkeypatch.setattr(index_build, "load_corpus", lambda **kwargs: documents)
    monkeypatch.setattr(index_build, "build_embeddings", lambda settings: embeddings)

    monkeypatch.setattr(
        index_build,
        "build_vector_store",
        lambda **kwargs: vector_store,
    )

    def fake_save_vector_store(**kwargs):
        assert not stale_file.exists()

    monkeypatch.setattr(
        index_build,
        "save_vector_store",
        fake_save_vector_store,
    )

    index_build.build_index(
        papers_dir = papers_dir,
        index_dir = index_dir,
        settings = settings,
        force = True,
    )

    assert not stale_file.exists()


def test_build_index_loads_settings_from_environment_when_not_provided(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        chunk_size = 700,
        chunk_overlap = 70,
    )

    documents = [
        Document(
            page_content = "Evidence.",
            metadata = {
                "chunk_id": 0,
                "source": "research.pdf",
                "page_number": 1,
            },
        )
    ]

    embeddings = object()
    vector_store = object()

    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        index_build.Settings,
        "from_env",
        classmethod(lambda cls: settings),
    )

    def fake_load_corpus(**kwargs):
        calls["load_corpus"] = kwargs
        return documents

    monkeypatch.setattr(index_build, "load_corpus", fake_load_corpus)
    monkeypatch.setattr(index_build, "build_embeddings", lambda settings: embeddings)
    monkeypatch.setattr(index_build, "build_vector_store", lambda **kwargs: vector_store)
    monkeypatch.setattr(index_build, "save_vector_store", lambda **kwargs: None)

    index_build.build_index(
        papers_dir = tmp_path / "papers",
        index_dir = tmp_path / "indexes",
    )

    assert calls["load_corpus"]["chunk_size"] == 700
    assert calls["load_corpus"]["chunk_overlap"] == 70
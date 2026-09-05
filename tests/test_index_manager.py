from pathlib import Path

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agentic_research_rag.config import Settings
from agentic_research_rag.retrieval.index_manager import (
    build_vector_store,
    load_or_build_vector_store,
)


class CountingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        text = text.lower()

        return [
            float("semantic" in text),
            float("lexical" in text),
            float("retrieval" in text),
        ]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        self.document_calls += 1

        return [
            self._vector(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        self.query_calls += 1

        return self._vector(text)


def make_documents() -> list[Document]:
    return [
        Document(
            page_content = (
                "Semantic retrieval uses dense vectors."
            ),
            metadata = {
                "chunk_id": 0,
                "source": "retrieval.pdf",
                "page_number": 1,
            },
        ),
        Document(
            page_content = (
                "Lexical retrieval uses exact terms."
            ),
            metadata = {
                "chunk_id": 1,
                "source": "retrieval.pdf",
                "page_number": 2,
            },
        ),
    ]


def test_build_vector_store_returns_faiss() -> None:
    embeddings = CountingEmbeddings()

    vector_store = build_vector_store(
        documents = make_documents(),
        embeddings = embeddings,
    )

    assert isinstance(
        vector_store,
        FAISS,
    )

    assert embeddings.document_calls == 1


def test_build_vector_store_requires_documents() -> None:
    embeddings = CountingEmbeddings()

    with pytest.raises(ValueError):
        build_vector_store(
            documents = [],
            embeddings = embeddings,
        )


def test_load_or_build_persists_index(
    tmp_path: Path,
) -> None:
    embeddings = CountingEmbeddings()
    settings = Settings()

    index_dir = tmp_path / "faiss"

    vector_store = load_or_build_vector_store(
        documents = make_documents(),
        embeddings = embeddings,
        settings = settings,
        index_dir = index_dir,
    )

    assert isinstance(
        vector_store,
        FAISS,
    )

    assert (
        index_dir / "index.faiss"
    ).exists()

    assert (
        index_dir / "index.pkl"
    ).exists()

    assert (
        index_dir / "metadata.json"
    ).exists()


def test_valid_index_is_loaded_without_reembedding(
    tmp_path: Path,
) -> None:
    embeddings = CountingEmbeddings()
    settings = Settings()

    index_dir = tmp_path / "faiss"
    documents = make_documents()

    load_or_build_vector_store(
        documents = documents,
        embeddings = embeddings,
        settings = settings,
        index_dir = index_dir,
    )

    assert embeddings.document_calls == 1

    load_or_build_vector_store(
        documents = documents,
        embeddings = embeddings,
        settings = settings,
        index_dir = index_dir,
    )

    assert embeddings.document_calls == 1


def test_changed_corpus_rebuilds_index(
    tmp_path: Path,
) -> None:
    embeddings = CountingEmbeddings()
    settings = Settings()

    index_dir = tmp_path / "faiss"

    documents = make_documents()

    load_or_build_vector_store(
        documents = documents,
        embeddings = embeddings,
        settings = settings,
        index_dir = index_dir,
    )

    modified_documents = make_documents()

    modified_documents[0].page_content = (
        "Semantic retrieval was modified."
    )

    load_or_build_vector_store(
        documents = modified_documents,
        embeddings = embeddings,
        settings = settings,
        index_dir = index_dir,
    )

    assert embeddings.document_calls == 2


def test_changed_embedding_model_rebuilds_index(
    tmp_path: Path,
) -> None:
    embeddings = CountingEmbeddings()

    first_settings = Settings(
        local_embedding_model = "model-a",
    )

    second_settings = Settings(
        local_embedding_model = "model-b",
    )

    index_dir = tmp_path / "faiss"
    documents = make_documents()

    load_or_build_vector_store(
        documents = documents,
        embeddings = embeddings,
        settings = first_settings,
        index_dir = index_dir,
    )

    load_or_build_vector_store(
        documents = documents,
        embeddings = embeddings,
        settings = second_settings,
        index_dir = index_dir,
    )

    assert embeddings.document_calls == 2


def test_vector_store_retrieves_documents() -> None:
    embeddings = CountingEmbeddings()

    vector_store = build_vector_store(
        documents = make_documents(),
        embeddings = embeddings,
    )

    results = vector_store.similarity_search(
        query = "semantic retrieval",
        k = 1,
    )

    assert len(results) == 1

    assert (
        results[0].metadata["chunk_id"]
        == 0
    )
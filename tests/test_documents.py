from pathlib import Path

import pytest
from langchain_core.documents import Document

from agentic_research_rag.ingestion.documents import (
    load_corpus,
    split_documents,
)


def test_split_documents_preserves_metadata() -> None:
    documents = [
        Document(
            page_content = (
                "Semantic retrieval represents text using dense vectors. "
                "Lexical retrieval instead relies on term matching. "
                "Hybrid retrieval combines both approaches."
            ),
            metadata = {
                "source": "retrieval.pdf",
                "document_name": "retrieval.pdf",
                "page_number": 3,
            },
        )
    ]

    chunks = split_documents(
        documents = documents,
        chunk_size = 80,
        chunk_overlap = 20,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.metadata["source"] == "retrieval.pdf"
        assert chunk.metadata["document_name"] == "retrieval.pdf"
        assert chunk.metadata["page_number"] == 3


def test_split_documents_returns_langchain_documents() -> None:
    documents = [
        Document(
            page_content = "A document about semantic retrieval.",
            metadata = {
                "source": "document.pdf",
            },
        )
    ]

    chunks = split_documents(
        documents = documents,
        chunk_size = 100,
        chunk_overlap = 20,
    )

    assert len(chunks) == 1
    assert isinstance(chunks[0], Document)
    assert chunks[0].page_content == (
        "A document about semantic retrieval."
    )


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_split_documents_validates_configuration(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        split_documents(
            documents = [],
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )


def test_load_corpus_requires_existing_directory(
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_corpus(
            papers_dir = missing_dir,
            chunk_size = 100,
            chunk_overlap = 20,
        )


def test_load_corpus_requires_directory(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("hello", encoding = "utf-8")

    with pytest.raises(ValueError):
        load_corpus(
            papers_dir = file_path,
            chunk_size = 100,
            chunk_overlap = 20,
        )


def test_load_corpus_requires_pdf_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        load_corpus(
            papers_dir = tmp_path,
            chunk_size = 100,
            chunk_overlap = 20,
        )
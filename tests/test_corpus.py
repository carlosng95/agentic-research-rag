import pytest

from agentic_research_rag.ingestion import corpus
from agentic_research_rag.types import Page


def test_load_corpus_loads_all_pdf_files(tmp_path, monkeypatch):
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"

    pdf_a.touch()
    pdf_b.touch()

    calls = []

    def fake_load_pdf(path):
        calls.append(path.name)

        return [
            Page(
                document_name = path.name,
                page_number = 1,
                text = f"content from {path.name}",
            )
        ]

    monkeypatch.setattr(
        corpus,
        "load_pdf",
        fake_load_pdf,
    )

    chunks = corpus.load_corpus(
        papers_dir = tmp_path,
        chunk_size = 100,
        overlap = 0,
    )

    assert calls == [
        "a.pdf",
        "b.pdf",
    ]

    assert len(chunks) == 2

    assert chunks[0].document_name == "a.pdf"
    assert chunks[1].document_name == "b.pdf"


def test_load_corpus_processes_pdfs_in_sorted_order(tmp_path, monkeypatch):
    (tmp_path / "z.pdf").touch()
    (tmp_path / "a.pdf").touch()
    (tmp_path / "m.pdf").touch()

    calls = []

    def fake_load_pdf(path):
        calls.append(path.name)

        return [
            Page(
                document_name = path.name,
                page_number = 1,
                text = path.name,
            )
        ]

    monkeypatch.setattr(
        corpus,
        "load_pdf",
        fake_load_pdf,
    )

    corpus.load_corpus(
        papers_dir = tmp_path,
        chunk_size = 100,
        overlap = 0,
    )

    assert calls == [
        "a.pdf",
        "m.pdf",
        "z.pdf",
    ]


def test_load_corpus_ignores_non_pdf_files(tmp_path, monkeypatch):
    (tmp_path / "paper.pdf").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "data.csv").touch()

    calls = []

    def fake_load_pdf(path):
        calls.append(path.name)

        return [
            Page(
                document_name = path.name,
                page_number = 1,
                text = "paper content",
            )
        ]

    monkeypatch.setattr(
        corpus,
        "load_pdf",
        fake_load_pdf,
    )

    chunks = corpus.load_corpus(
        papers_dir = tmp_path,
        chunk_size = 100,
        overlap = 0,
    )

    assert calls == [
        "paper.pdf",
    ]

    assert len(chunks) == 1
    assert chunks[0].document_name == "paper.pdf"


def test_load_corpus_combines_pages_from_multiple_pdfs(tmp_path, monkeypatch):
    (tmp_path / "paper_a.pdf").touch()
    (tmp_path / "paper_b.pdf").touch()

    def fake_load_pdf(path):
        if path.name == "paper_a.pdf":
            return [
                Page(
                    document_name = "paper_a.pdf",
                    page_number = 1,
                    text = "page one",
                ),
                Page(
                    document_name = "paper_a.pdf",
                    page_number = 2,
                    text = "page two",
                ),
            ]

        return [
            Page(
                document_name = "paper_b.pdf",
                page_number = 1,
                text = "page three",
            )
        ]

    monkeypatch.setattr(
        corpus,
        "load_pdf",
        fake_load_pdf,
    )

    chunks = corpus.load_corpus(
        papers_dir = tmp_path,
        chunk_size = 100,
        overlap = 0,
    )

    assert len(chunks) == 3

    assert chunks[0].document_name == "paper_a.pdf"
    assert chunks[0].page_number == 1

    assert chunks[1].document_name == "paper_a.pdf"
    assert chunks[1].page_number == 2

    assert chunks[2].document_name == "paper_b.pdf"
    assert chunks[2].page_number == 1


def test_load_corpus_assigns_global_consecutive_chunk_ids(tmp_path, monkeypatch):
    (tmp_path / "paper_a.pdf").touch()
    (tmp_path / "paper_b.pdf").touch()

    def fake_load_pdf(path):
        return [
            Page(
                document_name = path.name,
                page_number = 1,
                text = "abcdefgh",
            )
        ]

    monkeypatch.setattr(
        corpus,
        "load_pdf",
        fake_load_pdf,
    )

    chunks = corpus.load_corpus(
        papers_dir = tmp_path,
        chunk_size = 4,
        overlap = 0,
    )

    assert [
        chunk.chunk_id
        for chunk in chunks
    ] == [
        0,
        1,
        2,
        3,
    ]


def test_load_corpus_passes_chunk_configuration(tmp_path, monkeypatch):
    (tmp_path / "paper.pdf").touch()

    def fake_load_pdf(path):
        return [
            Page(
                document_name = path.name,
                page_number = 1,
                text = "abcdefghij",
            )
        ]

    monkeypatch.setattr(
        corpus,
        "load_pdf",
        fake_load_pdf,
    )

    chunks = corpus.load_corpus(
        papers_dir = tmp_path,
        chunk_size = 4,
        overlap = 2,
    )

    assert [
        chunk.text
        for chunk in chunks
    ] == [
        "abcd",
        "cdef",
        "efgh",
        "ghij",
    ]


def test_load_corpus_rejects_missing_directory(tmp_path):
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(
        FileNotFoundError,
        match = "Papers directory not found",
    ):
        corpus.load_corpus(
            papers_dir = missing_dir,
            chunk_size = 100,
            overlap = 20,
        )


def test_load_corpus_rejects_file_instead_of_directory(tmp_path):
    file_path = tmp_path / "paper.pdf"
    file_path.touch()

    with pytest.raises(
        ValueError,
        match = "Expected a directory",
    ):
        corpus.load_corpus(
            papers_dir = file_path,
            chunk_size = 100,
            overlap = 20,
        )


def test_load_corpus_rejects_directory_without_pdfs(tmp_path):
    (tmp_path / "notes.txt").touch()

    with pytest.raises(
        ValueError,
        match = "No PDF files found",
    ):
        corpus.load_corpus(
            papers_dir = tmp_path,
            chunk_size = 100,
            overlap = 20,
        )
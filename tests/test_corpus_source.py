from pathlib import Path

import pytest

from agentic_research_rag.config import Settings
from agentic_research_rag.ingestion.corpus_source import materialize_corpus_source


class FakePaginator:
    def __init__(self, pages: list[dict]) -> None:
        self.pages = pages
        self.calls: list[dict] = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)

        yield from self.pages


class FakeS3Client:
    def __init__(self, pages: list[dict]) -> None:
        self.paginator = FakePaginator(
            pages = pages,
        )

        self.downloads: list[dict] = []

    def get_paginator(self, operation_name: str):
        assert operation_name == "list_objects_v2"

        return self.paginator

    def download_file(
        self,
        Bucket: str,
        Key: str,
        Filename: str,
    ) -> None:
        self.downloads.append(
            {
                "Bucket": Bucket,
                "Key": Key,
                "Filename": Filename,
            }
        )

        Path(Filename).write_bytes(
            f"fake-pdf:{Key}".encode("utf-8")
        )


def test_materialize_local_corpus(tmp_path: Path) -> None:
    settings = Settings(
        corpus_source = "local",
    )

    with materialize_corpus_source(
        settings = settings,
        local_papers_dir = tmp_path,
    ) as corpus_dir:
        assert corpus_dir == tmp_path


def test_materialize_s3_corpus(tmp_path: Path) -> None:
    settings = Settings(
        corpus_source = "s3",
        corpus_bucket = "test-bucket",
        corpus_prefix = "research/source-documents",
    )

    s3_client = FakeS3Client(
        pages = [
            {
                "Contents": [
                    {
                        "Key": (
                            "research/source-documents/"
                            "group-a/paper-a.pdf"
                        )
                    },
                    {
                        "Key": (
                            "research/source-documents/"
                            "ignore.txt"
                        )
                    },
                ]
            },
            {
                "Contents": [
                    {
                        "Key": (
                            "research/source-documents/"
                            "group-b/paper-b.PDF"
                        )
                    }
                ]
            },
        ]
    )

    with materialize_corpus_source(
        settings = settings,
        s3_client = s3_client,
    ) as corpus_dir:
        assert (
            corpus_dir
            / "group-a"
            / "paper-a.pdf"
        ).is_file()

        assert (
            corpus_dir
            / "group-b"
            / "paper-b.PDF"
        ).is_file()

        assert not (
            corpus_dir
            / "ignore.txt"
        ).exists()

        assert len(s3_client.downloads) == 2

        assert s3_client.paginator.calls == [
            {
                "Bucket": "test-bucket",
                "Prefix": "research/source-documents/",
            }
        ]

        temporary_path = corpus_dir

    assert not temporary_path.exists()


def test_materialize_s3_corpus_requires_pdf_files() -> None:
    settings = Settings(
        corpus_source = "s3",
        corpus_bucket = "test-bucket",
        corpus_prefix = "research/source-documents",
    )

    s3_client = FakeS3Client(
        pages = [
            {
                "Contents": [
                    {
                        "Key": (
                            "research/source-documents/"
                            "notes.txt"
                        )
                    }
                ]
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match = "No PDF files found",
    ):
        with materialize_corpus_source(
            settings = settings,
            s3_client = s3_client,
        ):
            pass
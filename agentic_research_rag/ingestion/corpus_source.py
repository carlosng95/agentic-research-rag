from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import boto3

from ..config import Settings


def build_s3_client(settings: Settings) -> Any:
    return boto3.client(
        "s3",
        region_name = settings.aws_region,
    )


def _normalized_prefix(prefix: str) -> str:
    return prefix.strip("/")


def _relative_corpus_path(key: str, prefix: str) -> Path:
    prefix = _normalized_prefix(prefix)
    expected_prefix = f"{prefix}/"

    if not key.startswith(expected_prefix):
        raise ValueError(
            f"S3 object key is outside the configured corpus prefix: {key}"
        )

    relative_path = Path(
        key[len(expected_prefix):]
    )

    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or not relative_path.name
    ):
        raise ValueError(
            f"Invalid S3 corpus object key: {key}"
        )

    return relative_path


def _list_pdf_keys(
    settings: Settings,
    s3_client: Any,
) -> list[str]:
    prefix = _normalized_prefix(
        settings.corpus_prefix
    )

    paginator = s3_client.get_paginator(
        "list_objects_v2"
    )

    keys: list[str] = []

    for page in paginator.paginate(
        Bucket = settings.corpus_bucket,
        Prefix = f"{prefix}/",
    ):
        for item in page.get("Contents", []):
            key = str(
                item.get("Key", "")
            )

            if key.lower().endswith(".pdf"):
                keys.append(key)

    return sorted(keys)


@contextmanager
def materialize_corpus_source(
    settings: Settings,
    local_papers_dir: str | Path = "papers",
    s3_client: Any | None = None,
) -> Iterator[Path]:
    if settings.corpus_source == "local":
        yield Path(local_papers_dir)
        return

    if settings.corpus_source == "s3":
        if s3_client is None:
            s3_client = build_s3_client(
                settings = settings,
            )

        keys = _list_pdf_keys(
            settings = settings,
            s3_client = s3_client,
        )

        if not keys:
            raise ValueError(
                "No PDF files found in the configured S3 corpus."
            )

        with TemporaryDirectory(
            prefix = "agentic-rag-corpus-",
        ) as temporary_directory:
            corpus_dir = Path(
                temporary_directory
            )

            for key in keys:
                relative_path = _relative_corpus_path(
                    key = key,
                    prefix = settings.corpus_prefix,
                )

                destination = corpus_dir / relative_path

                destination.parent.mkdir(
                    parents = True,
                    exist_ok = True,
                )

                s3_client.download_file(
                    Bucket = settings.corpus_bucket,
                    Key = key,
                    Filename = str(destination),
                )

            yield corpus_dir

        return

    raise ValueError(
        f"Unsupported corpus source: {settings.corpus_source}"
    )
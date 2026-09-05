from pathlib import Path

from ..types import Chunk
from .chunker import chunk_pages
from .pdf_loader import load_pdf


def load_corpus(
    papers_dir: str | Path,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """
    Load all PDF documents from a directory and convert them into chunks.
    """

    papers_dir = Path(papers_dir)

    if not papers_dir.exists():
        raise FileNotFoundError(f"Papers directory not found: {papers_dir}")

    if not papers_dir.is_dir():
        raise ValueError(f"Expected a directory, got: {papers_dir}")

    pdf_files = sorted(papers_dir.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {papers_dir}")

    pages = []

    for pdf_path in pdf_files:
        pages.extend(load_pdf(pdf_path))

    return chunk_pages(
        pages = pages,
        chunk_size = chunk_size,
        overlap = overlap,
    )
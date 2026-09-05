from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_documents(pdf_path: str | Path) -> list[Document]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"Expected a file, got: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    loader = PyPDFLoader(
        file_path = str(pdf_path),
        mode = "page",
    )

    documents = loader.load()

    for document in documents:
        page = int(document.metadata.get("page", 0))

        document.metadata["source"] = pdf_path.name
        document.metadata["document_name"] = pdf_path.name
        document.metadata["page_number"] = page + 1

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        is_separator_regex = False,
    )

    return splitter.split_documents(documents)


def load_corpus(
    papers_dir: str | Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    papers_dir = Path(papers_dir)

    if not papers_dir.exists():
        raise FileNotFoundError(
            f"Papers directory not found: {papers_dir}"
        )

    if not papers_dir.is_dir():
        raise ValueError(
            f"Expected a directory, got: {papers_dir}"
        )

    pdf_files = sorted(papers_dir.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in: {papers_dir}"
        )

    documents: list[Document] = []

    for pdf_path in pdf_files:
        documents.extend(
            load_pdf_documents(pdf_path = pdf_path)
        )

    chunks = split_documents(
        documents = documents,
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id

    return chunks
from pathlib import Path

from pypdf import PdfReader

from ..types import Page


def load_pdf(pdf_path: str | Path) -> list[Page]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {pdf_path.suffix}"
        )

    reader = PdfReader(pdf_path)

    pages: list[Page] = []

    for page_number, pdf_page in enumerate(
        reader.pages,
        start=1,
    ):
        text = pdf_page.extract_text() or ""

        page = Page(
            document_name=pdf_path.name,
            page_number=page_number,
            text=text,
        )

        pages.append(page)

    return pages
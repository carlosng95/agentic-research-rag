from agentic_research_rag.ingestion import pdf_loader


class FakePDFPage:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class FakePDFReader:
    def __init__(self, path) -> None:
        self.pages = [
            FakePDFPage("First page text."),
            FakePDFPage("Second page text."),
        ]


def test_load_pdf_extracts_pages(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()

    monkeypatch.setattr(
        pdf_loader,
        "PdfReader",
        FakePDFReader,
    )

    pages = pdf_loader.load_pdf(
        pdf_path
    )

    assert len(pages) == 2

    assert pages[0].document_name == "paper.pdf"
    assert pages[0].page_number == 1
    assert pages[0].text == "First page text."

    assert pages[1].document_name == "paper.pdf"
    assert pages[1].page_number == 2
    assert pages[1].text == "Second page text."


def test_load_pdf_handles_page_without_text(monkeypatch, tmp_path):
    class ReaderWithEmptyPage:
        def __init__(self, path) -> None:
            self.pages = [
                FakePDFPage(None),
            ]

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()

    monkeypatch.setattr(
        pdf_loader,
        "PdfReader",
        ReaderWithEmptyPage,
    )

    pages = pdf_loader.load_pdf(
        pdf_path
    )

    assert len(pages) == 1
    assert pages[0].text == ""


def test_load_pdf_preserves_page_numbers(monkeypatch, tmp_path):
    class ThreePageReader:
        def __init__(self, path) -> None:
            self.pages = [
                FakePDFPage("Page A"),
                FakePDFPage("Page B"),
                FakePDFPage("Page C"),
            ]

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()

    monkeypatch.setattr(
        pdf_loader,
        "PdfReader",
        ThreePageReader,
    )

    pages = pdf_loader.load_pdf(
        pdf_path
    )

    assert [
        page.page_number
        for page in pages
    ] == [
        1,
        2,
        3,
    ]


def test_load_pdf_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing.pdf"

    try:
        pdf_loader.load_pdf(
            missing_path
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
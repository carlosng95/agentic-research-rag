from langchain_core.documents import Document
from langgraph.graph import MessagesState


class ResearchState(
    MessagesState,
    total = False,
):
    question: str
    standalone_query: str

    paper_documents: list[Document]
    paper_answer: str
    sufficient: bool

    web_documents: list[Document]

    final_documents: list[Document]
    final_answer: str

    citation_valid: bool
    citation_has_citations: bool
    cited_source_numbers: list[int]
    invalid_source_numbers: list[int]
    malformed_citations: list[str]

    tools_used: list[str]
    flags: list[str]
from ..rag_pipeline import RAGPipeline
from ..types import ResearchResponse
from .base import Tool


class PaperRAGTool(Tool):
    """
    Search and answer questions using the provided scientific papers.
    """

    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        self._rag_pipeline = rag_pipeline

    @property
    def name(self) -> str:
        return "paper_search"

    @property
    def description(self) -> str:
        return (
            "Search the provided scientific papers and answer questions "
            "using evidence retrieved from those papers."
        )

    def run(self, query: str) -> ResearchResponse:
        return self._rag_pipeline.answer(query = query)
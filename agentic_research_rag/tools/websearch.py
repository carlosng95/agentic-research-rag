from ..providers.websearch import WebSearchProvider
from ..types import ResearchResponse
from .base import Tool


class WebSearchTool(Tool):
    """
    Search the web for information not sufficiently covered
    by the provided scientific papers.
    """

    def __init__(self, web_search_provider: WebSearchProvider) -> None:
        self._web_search_provider = web_search_provider

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for external or up-to-date information "
            "when the scientific papers do not provide sufficient evidence."
        )

    def run(self, query: str) -> ResearchResponse:
        answer, sources = self._web_search_provider.search(
            query = query,
        )

        return ResearchResponse(
            answer = answer,
            sources = sources,
            reasoning_steps = [
                "Searched the web for additional evidence.",
            ],
            confidence = 0.0,
            flags = [],
        )
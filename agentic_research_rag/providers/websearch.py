from abc import ABC, abstractmethod

from ..config import get_required_env
from ..types import Source


class WebSearchUnavailableError(RuntimeError):
    """
    Raised when web search cannot be performed.
    """

    pass


class WebSearchProvider(ABC):
    """
    Interface implemented by web search providers.
    """

    @abstractmethod
    def search(self, query: str) -> tuple[str, list[Source]]:
        """
        Search the web and return an answer with its sources.
        """

        raise NotImplementedError


class OpenAIWebSearchProvider(WebSearchProvider):
    """
    Web search provider backed by the OpenAI Responses API.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        try:
            from openai import OpenAI

            self._model = model or get_required_env("LLM_MODEL")
            self._api_key = api_key or get_required_env("OPENAI_API_KEY")
            self._client = OpenAI(api_key = self._api_key)

        except Exception as error:
            raise WebSearchUnavailableError(
                f"Could not initialize OpenAI web search provider: {error}"
            ) from error

    @property
    def model_id(self) -> str:
        return self._model

    @staticmethod
    def _extract_sources(response) -> list[Source]:
        sources: list[Source] = []
        seen_urls: set[str] = set()

        for output_item in response.output:
            if getattr(output_item, "type", None) != "message":
                continue

            for content in output_item.content:
                if getattr(content, "type", None) != "output_text":
                    continue

                for annotation in content.annotations:
                    if getattr(annotation, "type", None) != "url_citation":
                        continue

                    url = annotation.url

                    if url in seen_urls:
                        continue

                    seen_urls.add(url)

                    sources.append(
                        Source(
                            type = "web",
                            ref = url,
                            locator = annotation.title,
                        )
                    )

        return sources

    def search(self, query: str) -> tuple[str, list[Source]]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        try:
            response = self._client.responses.create(
                model = self._model,
                tools = [{"type": "web_search"}],
                input = query,
            )

        except Exception as error:
            raise WebSearchUnavailableError(
                f"Web search failed: {error}"
            ) from error

        answer = response.output_text
        sources = self._extract_sources(response)

        return answer, sources
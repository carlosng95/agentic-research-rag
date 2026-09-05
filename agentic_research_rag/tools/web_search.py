from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch


def build_web_search_tool(
    max_results: int = 5,
) -> BaseTool:
    if max_results <= 0:
        raise ValueError(
            "max_results must be greater than 0."
        )

    return TavilySearch(
        max_results = max_results,
        topic = "general",
        search_depth = "basic",
        include_answer = False,
        include_raw_content = False,
        include_images = False,
    )


def web_results_to_documents(
    response: dict[str, Any],
) -> list[Document]:
    results = response.get(
        "results",
        [],
    )

    if not isinstance(results, list):
        raise TypeError(
            "Expected web search results to be a list."
        )

    documents: list[Document] = []
    seen_urls: set[str] = set()

    for rank, result in enumerate(
        results,
        start = 1,
    ):
        if not isinstance(result, dict):
            continue

        url = str(
            result.get(
                "url",
                "",
            )
        ).strip()

        title = str(
            result.get(
                "title",
                "",
            )
        ).strip()

        content = str(
            result.get(
                "content",
                "",
            )
        ).strip()

        if not url or not content:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        metadata: dict[str, Any] = {
            "source": url,
            "url": url,
            "title": title or url,
            "web_rank": rank,
        }

        score = result.get(
            "score"
        )

        if isinstance(
            score,
            (int, float),
        ):
            metadata["search_score"] = float(
                score
            )

        documents.append(
            Document(
                page_content = content,
                metadata = metadata,
            )
        )

    return documents


def search_web(
    query: str,
    tool: BaseTool,
) -> list[Document]:
    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    response = tool.invoke(
        {
            "query": query,
        }
    )

    if not isinstance(response, dict):
        raise TypeError(
            "Expected web search response to be a dictionary."
        )

    return web_results_to_documents(
        response = response
    )
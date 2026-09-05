from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from agentic_research_rag.chains.sufficiency import (
    SufficiencyResult,
)
from agentic_research_rag.graph.research_graph import (
    build_research_graph,
)


def make_paper_document() -> Document:
    return Document(
        page_content = "Local evidence.",
        metadata = {
            "chunk_id": 1,
            "source": "local.pdf",
            "page_number": 2,
        },
    )


def make_web_tool(
    calls: dict[str, int],
) -> StructuredTool:
    def fake_web_search(
        query: str,
    ) -> dict[str, Any]:
        calls["web"] += 1

        return {
            "results": [
                {
                    "title": "Web article",
                    "url": (
                        "https://example.com/article"
                    ),
                    "content": (
                        "Complementary web evidence."
                    ),
                    "score": 0.9,
                }
            ]
        }

    return StructuredTool.from_function(
        func = fake_web_search,
        name = "fake_web_search",
        description = "Fake web search.",
    )


def test_sufficient_paper_evidence_skips_web() -> None:
    paper_document = make_paper_document()

    web_calls = {
        "web": 0,
    }

    query_rewriter = RunnableLambda(
        lambda state: state["question"]
    )

    rag_chain = RunnableLambda(
        lambda query: {
            "answer": (
                "Local answer [SOURCE 1]."
            ),
            "documents": [
                paper_document
            ],
        }
    )

    sufficiency_chain = RunnableLambda(
        lambda state: SufficiencyResult(
            sufficient = True
        )
    )

    synthesis_chain = RunnableLambda(
        lambda state: {
            "answer": (
                "This should never be used."
            ),
            "documents": [],
        }
    )

    graph = build_research_graph(
        query_rewriter = query_rewriter,
        rag_chain = rag_chain,
        sufficiency_chain = sufficiency_chain,
        web_search_tool = make_web_tool(
            calls = web_calls
        ),
        synthesis_chain = synthesis_chain,
    )

    result = graph.invoke(
        {
            "question": "Research question",
            "messages": [],
        }
    )

    assert result["sufficient"] is True

    assert result["final_answer"] == (
        "Local answer [SOURCE 1]."
    )

    assert result["final_documents"] == [
        paper_document
    ]

    assert result["tools_used"] == [
        "paper_retrieval"
    ]

    assert web_calls["web"] == 0

    assert result["citation_valid"] is True

    assert (
        result["cited_source_numbers"]
        == [1]
    )

    assert len(
        result["messages"]
    ) == 2


def test_insufficient_paper_evidence_uses_web_and_synthesis() -> None:
    paper_document = make_paper_document()

    web_calls = {
        "web": 0,
    }

    query_rewriter = RunnableLambda(
        lambda state: state["question"]
    )

    rag_chain = RunnableLambda(
        lambda query: {
            "answer": (
                "Local evidence is insufficient."
            ),
            "documents": [
                paper_document
            ],
        }
    )

    sufficiency_chain = RunnableLambda(
        lambda state: SufficiencyResult(
            sufficient = False
        )
    )

    def synthesize(
        state: dict,
    ) -> dict:
        documents = (
            state["paper_documents"]
            + state["web_documents"]
        )

        return {
            "answer": (
                "Combined answer "
                "[SOURCE 1] [SOURCE 2]."
            ),
            "documents": documents,
        }

    synthesis_chain = RunnableLambda(
        synthesize
    )

    graph = build_research_graph(
        query_rewriter = query_rewriter,
        rag_chain = rag_chain,
        sufficiency_chain = sufficiency_chain,
        web_search_tool = make_web_tool(
            calls = web_calls
        ),
        synthesis_chain = synthesis_chain,
    )

    result = graph.invoke(
        {
            "question": "Research question",
            "messages": [],
        }
    )

    assert result["sufficient"] is False

    assert web_calls["web"] == 1

    assert len(
        result["web_documents"]
    ) == 1

    assert len(
        result["final_documents"]
    ) == 2

    assert result["final_answer"] == (
        "Combined answer "
        "[SOURCE 1] [SOURCE 2]."
    )

    assert result["tools_used"] == [
        "paper_retrieval",
        "web_search",
    ]

    assert (
        "web_fallback_used"
        in result["flags"]
    )

    assert result["citation_valid"] is True

    assert (
        result["cited_source_numbers"]
        == [
            1,
            2,
        ]
    )


def test_checkpointer_preserves_conversation_history() -> None:
    paper_document = make_paper_document()

    histories: list[list[str]] = []

    def rewrite(
        state: dict,
    ) -> str:
        histories.append(
            [
                str(message.content)
                for message in state["history"]
            ]
        )

        return state["question"]

    query_rewriter = RunnableLambda(
        rewrite
    )

    rag_chain = RunnableLambda(
        lambda query: {
            "answer": (
                f"Answer to {query} [SOURCE 1]."
            ),
            "documents": [
                paper_document
            ],
        }
    )

    sufficiency_chain = RunnableLambda(
        lambda state: SufficiencyResult(
            sufficient = True
        )
    )

    synthesis_chain = RunnableLambda(
        lambda state: {
            "answer": "Unused.",
            "documents": [],
        }
    )

    graph = build_research_graph(
        query_rewriter = query_rewriter,
        rag_chain = rag_chain,
        sufficiency_chain = sufficiency_chain,
        web_search_tool = make_web_tool(
            calls = {
                "web": 0,
            }
        ),
        synthesis_chain = synthesis_chain,
        checkpointer = InMemorySaver(),
    )

    config = {
        "configurable": {
            "thread_id": "test-thread",
        }
    }

    first_result = graph.invoke(
        {
            "question": "First question",
            "messages": [],
        },
        config = config,
    )

    assert len(
        first_result["messages"]
    ) == 2

    second_result = graph.invoke(
        {
            "question": "Follow-up question",
            "messages": [],
        },
        config = config,
    )

    assert len(
        second_result["messages"]
    ) == 4

    assert histories[0] == []

    assert histories[1] == [
        "First question",
        (
            "Answer to First question "
            "[SOURCE 1]."
        ),
    ]
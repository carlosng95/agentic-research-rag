from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from agentic_research_rag.graph.state import (
    ResearchState,
)


def test_research_state_accumulates_messages() -> None:
    def add_user_message(
        state: ResearchState,
    ) -> dict:
        return {
            "messages": [
                HumanMessage(
                    content = "Research question"
                )
            ]
        }

    def add_assistant_message(
        state: ResearchState,
    ) -> dict:
        return {
            "messages": [
                AIMessage(
                    content = "Research answer"
                )
            ]
        }

    builder = StateGraph(
        ResearchState
    )

    builder.add_node(
        "user",
        add_user_message,
    )

    builder.add_node(
        "assistant",
        add_assistant_message,
    )

    builder.add_edge(
        START,
        "user",
    )

    builder.add_edge(
        "user",
        "assistant",
    )

    builder.add_edge(
        "assistant",
        END,
    )

    graph = builder.compile()

    result = graph.invoke(
        {
            "question": "Research question",
            "messages": [],
        }
    )

    assert len(
        result["messages"]
    ) == 2

    assert isinstance(
        result["messages"][0],
        HumanMessage,
    )

    assert isinstance(
        result["messages"][1],
        AIMessage,
    )

    assert (
        result["messages"][0].content
        == "Research question"
    )

    assert (
        result["messages"][1].content
        == "Research answer"
    )


def test_research_state_preserves_other_fields() -> None:
    def rewrite(
        state: ResearchState,
    ) -> dict:
        return {
            "standalone_query": (
                "Standalone research query"
            )
        }

    builder = StateGraph(
        ResearchState
    )

    builder.add_node(
        "rewrite",
        rewrite,
    )

    builder.add_edge(
        START,
        "rewrite",
    )

    builder.add_edge(
        "rewrite",
        END,
    )

    graph = builder.compile()

    result = graph.invoke(
        {
            "question": "Original question",
            "messages": [],
        }
    )

    assert (
        result["question"]
        == "Original question"
    )

    assert (
        result["standalone_query"]
        == "Standalone research query"
    )
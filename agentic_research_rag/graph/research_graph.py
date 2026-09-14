from typing import Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.graph.state import CompiledStateGraph

from ..citation_validator import validate_citations
from ..observability.nodes import observe_node
from ..tools.web_search import search_web
from .state import ResearchState


def _append_unique(
    values: list[str],
    value: str,
) -> list[str]:
    if value in values:
        return values

    return [
        *values,
        value,
    ]


def build_research_graph(
    query_rewriter: Runnable,
    rag_chain: Runnable,
    sufficiency_chain: Runnable,
    web_search_tool: BaseTool,
    synthesis_chain: Runnable,
    memory_turns: int = 5,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    if memory_turns < 0:
        raise ValueError(
            "memory_turns must be greater than or equal to 0."
        )

    def begin_turn(
        state: ResearchState,
    ) -> dict:
        question = state["question"]

        return {
            "messages": [
                HumanMessage(
                    content = question
                )
            ],
            "standalone_query": "",
            "paper_documents": [],
            "paper_answer": "",
            "sufficient": False,
            "web_documents": [],
            "final_documents": [],
            "final_answer": "",
            "citation_valid": True,
            "citation_has_citations": False,
            "cited_source_numbers": [],
            "invalid_source_numbers": [],
            "malformed_citations": [],
            "tools_used": [],
            "flags": [],
        }

    def rewrite_query(
        state: ResearchState,
    ) -> dict:
        messages: list[BaseMessage] = list(
            state.get(
                "messages",
                [],
            )
        )

        history = messages[:-1]

        if memory_turns == 0:
            history = []
        else:
            history = history[
                -(memory_turns * 2):
            ]

        standalone_query = query_rewriter.invoke(
            {
                "question": state["question"],
                "history": history,
            }
        )

        return {
            "standalone_query": standalone_query,
        }

    def paper_rag(
        state: ResearchState,
    ) -> dict:
        result = rag_chain.invoke(
            state["standalone_query"]
        )

        return {
            "paper_documents": result["documents"],
            "paper_answer": result["answer"],
            "tools_used": [
                "paper_retrieval",
            ],
        }

    def evaluate(
        state: ResearchState,
    ) -> dict:
        result = sufficiency_chain.invoke(
            {
                "question": state["standalone_query"],
                "answer": state["paper_answer"],
                "documents": state["paper_documents"],
            }
        )

        return {
            "sufficient": result.sufficient,
        }

    def route_after_evaluation(
        state: ResearchState,
    ) -> Literal[
        "finalize_paper",
        "web_search",
    ]:
        if state["sufficient"]:
            return "finalize_paper"

        return "web_search"

    def finalize_paper(
        state: ResearchState,
    ) -> dict:
        return {
            "final_answer": state["paper_answer"],
            "final_documents": state[
                "paper_documents"
            ],
        }

    def web_search(
        state: ResearchState,
    ) -> dict:
        documents = search_web(
            query = state["standalone_query"],
            tool = web_search_tool,
        )

        tools_used = _append_unique(
            values = list(
                state.get(
                    "tools_used",
                    [],
                )
            ),
            value = "web_search",
        )

        flags = _append_unique(
            values = list(
                state.get(
                    "flags",
                    [],
                )
            ),
            value = "web_fallback_used",
        )

        if not documents:
            flags = _append_unique(
                values = flags,
                value = "web_search_no_results",
            )

        return {
            "web_documents": documents,
            "tools_used": tools_used,
            "flags": flags,
        }

    def synthesize(
        state: ResearchState,
    ) -> dict:
        result = synthesis_chain.invoke(
            {
                "question": state["standalone_query"],
                "paper_documents": state[
                    "paper_documents"
                ],
                "web_documents": state[
                    "web_documents"
                ],
            }
        )

        return {
            "final_answer": result["answer"],
            "final_documents": result["documents"],
        }

    def validate_final_citations(
        state: ResearchState,
    ) -> dict:
        result = validate_citations(
            answer = state["final_answer"],
            documents = state[
                "final_documents"
            ],
        )

        flags = list(
            state.get(
                "flags",
                [],
            )
        )

        if not result.valid:
            flags = _append_unique(
                values = flags,
                value = "citations_invalid",
            )

        if (
            state["final_documents"]
            and not result.has_citations
        ):
            flags = _append_unique(
                values = flags,
                value = "citations_missing",
            )

        if not state["final_documents"]:
            flags = _append_unique(
                values = flags,
                value = "evidence_insufficient",
            )

        return {
            "citation_valid": result.valid,
            "citation_has_citations": (
                result.has_citations
            ),
            "cited_source_numbers": (
                result.cited_source_numbers
            ),
            "invalid_source_numbers": (
                result.invalid_source_numbers
            ),
            "malformed_citations": (
                result.malformed_citations
            ),
            "flags": flags,
        }

    def store_assistant_message(
        state: ResearchState,
    ) -> dict:
        return {
            "messages": [
                AIMessage(
                    content = state[
                        "final_answer"
                    ]
                )
            ]
        }

    builder = StateGraph(
        ResearchState
    )

    builder.add_node(
        "begin_turn",
        observe_node(
            "begin_turn",
            begin_turn,
        ),
    )

    builder.add_node(
        "rewrite_query",
        observe_node(
            "rewrite_query",
            rewrite_query,
        ),
    )

    builder.add_node(
        "paper_rag",
        observe_node(
            "paper_rag",
            paper_rag,
        ),
    )

    builder.add_node(
        "evaluate",
        observe_node(
            "evaluate",
            evaluate,
        ),
    )

    builder.add_node(
        "finalize_paper",
        observe_node(
            "finalize_paper",
            finalize_paper,
        ),
    )

    builder.add_node(
        "web_search",
        observe_node(
            "web_search",
            web_search,
        ),
    )

    builder.add_node(
        "synthesize",
        observe_node(
            "synthesize",
            synthesize,
        ),
    )

    builder.add_node(
        "validate_citations",
        observe_node(
            "validate_citations",
            validate_final_citations,
        ),
    )

    builder.add_node(
        "store_assistant_message",
        observe_node(
            "store_assistant_message",
            store_assistant_message,
        ),
    )

    builder.add_edge(
        START,
        "begin_turn",
    )

    builder.add_edge(
        "begin_turn",
        "rewrite_query",
    )

    builder.add_edge(
        "rewrite_query",
        "paper_rag",
    )

    builder.add_edge(
        "paper_rag",
        "evaluate",
    )

    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
    )

    builder.add_edge(
        "finalize_paper",
        "validate_citations",
    )

    builder.add_edge(
        "web_search",
        "synthesize",
    )

    builder.add_edge(
        "synthesize",
        "validate_citations",
    )

    builder.add_edge(
        "validate_citations",
        "store_assistant_message",
    )

    builder.add_edge(
        "store_assistant_message",
        END,
    )

    return builder.compile(
        checkpointer = checkpointer,
        name = "agentic-research-rag",
    )
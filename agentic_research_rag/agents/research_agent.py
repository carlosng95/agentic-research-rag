from ..evaluation.sufficiency import SufficiencyEvaluator
from ..memory import ConversationMemory
from ..query_rewriter import QueryRewriter
from ..synthesis.evidence import EvidenceSynthesizer
from ..tools.paper_rag import PaperRAGTool
from ..tools.websearch import WebSearchTool
from ..types import ResearchResponse


class ResearchAgent:
    """
    Research agent that searches papers first, optionally uses
    web search, and maintains conversational context.
    """

    def __init__(
        self,
        paper_tool: PaperRAGTool,
        web_tool: WebSearchTool,
        sufficiency_evaluator: SufficiencyEvaluator,
        synthesizer: EvidenceSynthesizer,
        memory: ConversationMemory,
        query_rewriter: QueryRewriter,
    ) -> None:
        self._paper_tool = paper_tool
        self._web_tool = web_tool
        self._sufficiency_evaluator = sufficiency_evaluator
        self._synthesizer = synthesizer
        self._memory = memory
        self._query_rewriter = query_rewriter

    def run(self, query: str) -> ResearchResponse:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        standalone_query = self._query_rewriter.rewrite(
            query = query,
            memory = self._memory,
        )

        paper_response = self._paper_tool.run(
            query = standalone_query,
        )

        paper_response.tools_used = [
            self._paper_tool.name,
        ]

        sufficient = self._sufficiency_evaluator.is_sufficient(
            query = standalone_query,
            response = paper_response,
        )

        if sufficient:
            final_response = paper_response

        else:
            web_response = self._web_tool.run(
                query = standalone_query,
            )

            final_response = self._synthesizer.synthesize(
                query = standalone_query,
                paper_response = paper_response,
                web_response = web_response,
            )

            final_response.tools_used = [
                self._paper_tool.name,
                self._web_tool.name,
            ]

        self._memory.add_user(query)
        self._memory.add_assistant(final_response.answer)

        return final_response
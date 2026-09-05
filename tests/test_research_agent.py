import pytest

from agentic_research_rag.agents.research_agent import ResearchAgent
from agentic_research_rag.memory import ConversationMemory
from agentic_research_rag.types import ResearchResponse, Source


class FakeTool:
    def __init__(self, name: str, response: ResearchResponse) -> None:
        self._name = name
        self._response = response
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def run(self, query: str) -> ResearchResponse:
        self.calls.append(query)
        return self._response


class FakeSufficiencyEvaluator:
    def __init__(self, result: bool) -> None:
        self._result = result
        self.calls: list[dict] = []

    def is_sufficient(self, query: str, response: ResearchResponse) -> bool:
        self.calls.append(
            {
                "query": query,
                "response": response,
            }
        )

        return self._result


class FakeSynthesizer:
    def __init__(self, response: ResearchResponse) -> None:
        self._response = response
        self.calls: list[dict] = []

    def synthesize(
        self,
        query: str,
        paper_response: ResearchResponse,
        web_response: ResearchResponse,
    ) -> ResearchResponse:
        self.calls.append(
            {
                "query": query,
                "paper_response": paper_response,
                "web_response": web_response,
            }
        )

        return self._response


class FakeQueryRewriter:
    def __init__(self, rewritten_query: str) -> None:
        self._rewritten_query = rewritten_query
        self.calls: list[dict] = []

    def rewrite(self, query: str, memory: ConversationMemory) -> str:
        self.calls.append(
            {
                "query": query,
                "memory": memory,
            }
        )

        return self._rewritten_query


def make_paper_response() -> ResearchResponse:
    return ResearchResponse(
        answer = "Hybrid retrieval combines semantic and lexical search.",
        sources = [
            Source(
                type = "paper",
                ref = "paper.pdf",
                locator = "page 2",
                snippet = (
                    "Semantic and lexical retrieval provide "
                    "complementary ranking signals."
                ),
            ),
        ],
    )


def make_web_response() -> ResearchResponse:
    return ResearchResponse(
        answer = "Web search result.",
        sources = [
            Source(
                type = "web",
                ref = "https://example.com",
                locator = "Example",
            ),
        ],
    )


def make_synthesized_response() -> ResearchResponse:
    return ResearchResponse(
        answer = "Combined paper and web answer.",
        sources = [
            Source(
                type = "paper",
                ref = "paper.pdf",
                locator = "page 2",
            ),
            Source(
                type = "web",
                ref = "https://example.com",
                locator = "Example",
            ),
        ],
    )


def test_agent_returns_paper_response_when_evidence_is_sufficient():
    paper_response = make_paper_response()

    paper_tool = FakeTool(
        name = "paper_search",
        response = paper_response,
    )

    web_tool = FakeTool(
        name = "web_search",
        response = make_web_response(),
    )

    evaluator = FakeSufficiencyEvaluator(
        result = True,
    )

    synthesizer = FakeSynthesizer(
        response = make_synthesized_response(),
    )

    memory = ConversationMemory(
        max_turns = 5,
    )

    query_rewriter = FakeQueryRewriter(
        rewritten_query = "Standalone retrieval query",
    )

    agent = ResearchAgent(
        paper_tool = paper_tool,
        web_tool = web_tool,
        sufficiency_evaluator = evaluator,
        synthesizer = synthesizer,
        memory = memory,
        query_rewriter = query_rewriter,
    )

    response = agent.run(
        query = "How does hybrid retrieval work?"
    )

    assert response is paper_response
    assert response.tools_used == ["paper_search"]

    assert paper_tool.calls == [
        "Standalone retrieval query",
    ]

    assert web_tool.calls == []
    assert synthesizer.calls == []


def test_agent_uses_web_when_paper_evidence_is_insufficient():
    paper_response = make_paper_response()
    web_response = make_web_response()
    synthesized_response = make_synthesized_response()

    paper_tool = FakeTool(
        name = "paper_search",
        response = paper_response,
    )

    web_tool = FakeTool(
        name = "web_search",
        response = web_response,
    )

    evaluator = FakeSufficiencyEvaluator(
        result = False,
    )

    synthesizer = FakeSynthesizer(
        response = synthesized_response,
    )

    memory = ConversationMemory(
        max_turns = 5,
    )

    query_rewriter = FakeQueryRewriter(
        rewritten_query = "Standalone external query",
    )

    agent = ResearchAgent(
        paper_tool = paper_tool,
        web_tool = web_tool,
        sufficiency_evaluator = evaluator,
        synthesizer = synthesizer,
        memory = memory,
        query_rewriter = query_rewriter,
    )

    response = agent.run(
        query = "What about the latest information?"
    )

    assert response is synthesized_response

    assert paper_tool.calls == [
        "Standalone external query",
    ]

    assert web_tool.calls == [
        "Standalone external query",
    ]

    assert len(synthesizer.calls) == 1

    assert synthesizer.calls[0]["query"] == "Standalone external query"
    assert synthesizer.calls[0]["paper_response"] is paper_response
    assert synthesizer.calls[0]["web_response"] is web_response

    assert response.tools_used == [
        "paper_search",
        "web_search",
    ]


def test_agent_passes_standalone_query_to_sufficiency_evaluator():
    paper_response = make_paper_response()

    paper_tool = FakeTool(
        name = "paper_search",
        response = paper_response,
    )

    web_tool = FakeTool(
        name = "web_search",
        response = make_web_response(),
    )

    evaluator = FakeSufficiencyEvaluator(
        result = True,
    )

    synthesizer = FakeSynthesizer(
        response = make_synthesized_response(),
    )

    memory = ConversationMemory(
        max_turns = 5,
    )

    query_rewriter = FakeQueryRewriter(
        rewritten_query = "Resolved standalone query",
    )

    agent = ResearchAgent(
        paper_tool = paper_tool,
        web_tool = web_tool,
        sufficiency_evaluator = evaluator,
        synthesizer = synthesizer,
        memory = memory,
        query_rewriter = query_rewriter,
    )

    agent.run(
        query = "And how does reranking improve it?"
    )

    assert len(evaluator.calls) == 1
    assert evaluator.calls[0]["query"] == "Resolved standalone query"
    assert evaluator.calls[0]["response"] is paper_response


def test_agent_stores_original_query_in_memory():
    paper_response = make_paper_response()

    paper_tool = FakeTool(
        name = "paper_search",
        response = paper_response,
    )

    web_tool = FakeTool(
        name = "web_search",
        response = make_web_response(),
    )

    evaluator = FakeSufficiencyEvaluator(
        result = True,
    )

    synthesizer = FakeSynthesizer(
        response = make_synthesized_response(),
    )

    memory = ConversationMemory(
        max_turns = 5,
    )

    query_rewriter = FakeQueryRewriter(
        rewritten_query = "Rewritten standalone query",
    )

    agent = ResearchAgent(
        paper_tool = paper_tool,
        web_tool = web_tool,
        sufficiency_evaluator = evaluator,
        synthesizer = synthesizer,
        memory = memory,
        query_rewriter = query_rewriter,
    )

    original_query = "And how does reranking improve it?"

    agent.run(
        query = original_query,
    )

    assert len(memory.turns) == 2

    assert memory.turns[0].role == "user"
    assert memory.turns[0].content == original_query

    assert memory.turns[1].role == "assistant"
    assert memory.turns[1].content == paper_response.answer


def test_agent_passes_memory_to_query_rewriter():
    paper_response = make_paper_response()

    paper_tool = FakeTool(
        name = "paper_search",
        response = paper_response,
    )

    web_tool = FakeTool(
        name = "web_search",
        response = make_web_response(),
    )

    evaluator = FakeSufficiencyEvaluator(
        result = True,
    )

    synthesizer = FakeSynthesizer(
        response = make_synthesized_response(),
    )

    memory = ConversationMemory(
        max_turns = 5,
    )

    memory.add_user(
        "Previous question"
    )

    memory.add_assistant(
        "Previous answer"
    )

    query_rewriter = FakeQueryRewriter(
        rewritten_query = "Standalone query",
    )

    agent = ResearchAgent(
        paper_tool = paper_tool,
        web_tool = web_tool,
        sufficiency_evaluator = evaluator,
        synthesizer = synthesizer,
        memory = memory,
        query_rewriter = query_rewriter,
    )

    agent.run(
        query = "Follow-up question"
    )

    assert len(query_rewriter.calls) == 1
    assert query_rewriter.calls[0]["query"] == "Follow-up question"
    assert query_rewriter.calls[0]["memory"] is memory


def test_agent_stores_synthesized_answer_in_memory_after_web_fallback():
    paper_tool = FakeTool(
        name = "paper_search",
        response = make_paper_response(),
    )

    web_tool = FakeTool(
        name = "web_search",
        response = make_web_response(),
    )

    evaluator = FakeSufficiencyEvaluator(
        result = False,
    )

    synthesized_response = make_synthesized_response()

    synthesizer = FakeSynthesizer(
        response = synthesized_response,
    )

    memory = ConversationMemory(
        max_turns = 5,
    )

    query_rewriter = FakeQueryRewriter(
        rewritten_query = "Standalone query",
    )

    agent = ResearchAgent(
        paper_tool = paper_tool,
        web_tool = web_tool,
        sufficiency_evaluator = evaluator,
        synthesizer = synthesizer,
        memory = memory,
        query_rewriter = query_rewriter,
    )

    agent.run(
        query = "Question requiring web search"
    )

    assert len(memory.turns) == 2
    assert memory.turns[1].role == "assistant"
    assert memory.turns[1].content == synthesized_response.answer


def test_agent_rejects_empty_query_without_executing_tools():
    paper_tool = FakeTool(
        name = "paper_search",
        response = make_paper_response(),
    )

    web_tool = FakeTool(
        name = "web_search",
        response = make_web_response(),
    )

    evaluator = FakeSufficiencyEvaluator(
        result = True,
    )

    synthesizer = FakeSynthesizer(
        response = make_synthesized_response(),
    )

    memory = ConversationMemory(
        max_turns = 5,
    )

    query_rewriter = FakeQueryRewriter(
        rewritten_query = "Anything",
    )

    agent = ResearchAgent(
        paper_tool = paper_tool,
        web_tool = web_tool,
        sufficiency_evaluator = evaluator,
        synthesizer = synthesizer,
        memory = memory,
        query_rewriter = query_rewriter,
    )

    with pytest.raises(
        ValueError,
        match = "Query cannot be empty",
    ):
        agent.run(
            query = "   "
        )

    assert paper_tool.calls == []
    assert web_tool.calls == []
    assert evaluator.calls == []
    assert synthesizer.calls == []
    assert query_rewriter.calls == []
    assert memory.turns == []

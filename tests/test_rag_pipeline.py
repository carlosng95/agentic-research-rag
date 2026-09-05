import pytest

from agentic_research_rag.rag_pipeline import RAGPipeline
from agentic_research_rag.types import Chunk
from tests.fakes import FakeLLMProvider


class FakeRetrievalPipeline:
    def __init__(self, results: list[Chunk]) -> None:
        self._results = results
        self.calls: list[str] = []

    def search(self, query: str) -> list[Chunk]:
        self.calls.append(query)
        return self._results


def make_chunk(
    chunk_id: int,
    document_name: str = "paper.pdf",
    page_number: int = 1,
    text: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id = chunk_id,
        document_name = document_name,
        page_number = page_number,
        text = text or f"Evidence from chunk {chunk_id}.",
        score = 1.0,
    )


def test_rag_pipeline_passes_query_to_retrieval():
    chunks = [
        make_chunk(1),
    ]

    retrieval_pipeline = FakeRetrievalPipeline(
        results = chunks,
    )

    llm = FakeLLMProvider(
        response = "Answer supported by evidence [SOURCE 1]."
    )

    rag = RAGPipeline(
        retrieval_pipeline = retrieval_pipeline,
        llm = llm,
    )

    query = "What is semantic search?"

    rag.answer(
        query = query,
    )

    assert retrieval_pipeline.calls == [
        query,
    ]


def test_rag_pipeline_builds_prompt_with_query_and_chunk_evidence():
    chunks = [
        make_chunk(
            chunk_id = 1,
            document_name = "retrieval_systems.pdf",
            page_number = 3,
            text = "Hybrid retrieval combines lexical and semantic search.",
        ),
    ]

    retrieval_pipeline = FakeRetrievalPipeline(
        results = chunks,
    )

    llm = FakeLLMProvider(
        response = "Hybrid retrieval combines lexical and semantic signals [SOURCE 1]."
    )

    rag = RAGPipeline(
        retrieval_pipeline = retrieval_pipeline,
        llm = llm,
    )

    query = "How does hybrid retrieval work?"

    rag.answer(
        query = query,
    )

    assert len(llm.prompts) == 1

    prompt = llm.prompts[0]

    assert query in prompt
    assert "Hybrid retrieval combines lexical and semantic search." in prompt
    assert "retrieval_systems.pdf" in prompt
    assert "Page: 3" in prompt
    assert "[SOURCE 1]" in prompt


def test_rag_pipeline_returns_generated_answer():
    chunks = [
        make_chunk(1),
    ]

    retrieval_pipeline = FakeRetrievalPipeline(
        results = chunks,
    )

    generated_answer = (
        "Semantic search retrieves information using vector similarity "
        "[SOURCE 1]."
    )

    llm = FakeLLMProvider(
        response = generated_answer,
    )

    rag = RAGPipeline(
        retrieval_pipeline = retrieval_pipeline,
        llm = llm,
    )

    response = rag.answer(
        query = "How does semantic search work?"
    )

    assert response.answer == generated_answer


def test_rag_pipeline_converts_chunks_to_sources():
    chunks = [
        make_chunk(
            chunk_id = 10,
            document_name = "paper_a.pdf",
            page_number = 2,
            text = "Evidence A",
        ),
        make_chunk(
            chunk_id = 20,
            document_name = "paper_b.pdf",
            page_number = 7,
            text = "Evidence B",
        ),
    ]

    retrieval_pipeline = FakeRetrievalPipeline(
        results = chunks,
    )

    llm = FakeLLMProvider(
        response = "Combined answer [SOURCE 1][SOURCE 2]."
    )

    rag = RAGPipeline(
        retrieval_pipeline = retrieval_pipeline,
        llm = llm,
    )

    response = rag.answer(
        query = "What does the evidence show?"
    )

    assert len(response.sources) == 2

    source_1 = response.sources[0]

    assert source_1.type == "paper"
    assert source_1.ref == "paper_a.pdf"
    assert source_1.locator == "page 2"
    assert source_1.snippet == "Evidence A"

    source_2 = response.sources[1]

    assert source_2.type == "paper"
    assert source_2.ref == "paper_b.pdf"
    assert source_2.locator == "page 7"
    assert source_2.snippet == "Evidence B"


def test_rag_pipeline_returns_no_results_response_without_calling_llm():
    retrieval_pipeline = FakeRetrievalPipeline(
        results = [],
    )

    llm = FakeLLMProvider(
        response = "This should never be generated."
    )

    rag = RAGPipeline(
        retrieval_pipeline = retrieval_pipeline,
        llm = llm,
    )

    response = rag.answer(
        query = "Question without evidence"
    )

    assert response.answer == (
        "No relevant information was found in the available papers."
    )

    assert response.sources == []
    assert response.flags == ["no_retrieval_results"]
    assert llm.prompts == []


def test_rag_pipeline_detects_missing_citations():
    chunks = [
        make_chunk(1),
    ]

    retrieval_pipeline = FakeRetrievalPipeline(
        results = chunks,
    )

    llm = FakeLLMProvider(
        response = "Semantic retrieval uses dense vector representations."
    )

    rag = RAGPipeline(
        retrieval_pipeline = retrieval_pipeline,
        llm = llm,
    )

    response = rag.answer(
        query = "How does semantic retrieval work?"
    )

    assert response.flags == [
        "missing_citations",
    ]


def test_rag_pipeline_detects_invalid_citation():
    chunks = [
        make_chunk(1),
        make_chunk(2),
    ]

    retrieval_pipeline = FakeRetrievalPipeline(
        results = chunks,
    )

    llm = FakeLLMProvider(
        response = "Semantic search relies on vector similarity [SOURCE 8]."
    )

    rag = RAGPipeline(
        retrieval_pipeline = retrieval_pipeline,
        llm = llm,
    )

    response = rag.answer(
        query = "What is semantic search?"
    )

    assert response.flags == [
        "invalid_source_8",
    ]


def test_rag_pipeline_accepts_valid_citations():
    chunks = [
        make_chunk(1),
        make_chunk(2),
    ]

    retrieval_pipeline = FakeRetrievalPipeline(
        results = chunks,
    )

    llm = FakeLLMProvider(
        response = (
            "Semantic retrieval uses vector similarity [SOURCE 1] "
            "while lexical retrieval uses term matching [SOURCE 2]."
        )
    )

    rag = RAGPipeline(
        retrieval_pipeline = retrieval_pipeline,
        llm = llm,
    )

    response = rag.answer(
        query = "How do semantic and lexical retrieval differ?"
    )

    assert response.flags == []


def test_rag_pipeline_rejects_empty_query_without_running_pipeline():
    retrieval_pipeline = FakeRetrievalPipeline(
        results = [
            make_chunk(1),
        ],
    )

    llm = FakeLLMProvider(
        response = "This should never be generated."
    )

    rag = RAGPipeline(
        retrieval_pipeline = retrieval_pipeline,
        llm = llm,
    )

    with pytest.raises(
        ValueError,
        match = "Query cannot be empty",
    ):
        rag.answer(
            query = "   "
        )

    assert retrieval_pipeline.calls == []
    assert llm.prompts == []

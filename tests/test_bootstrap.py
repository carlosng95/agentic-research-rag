from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from agentic_research_rag import bootstrap
from agentic_research_rag.config import Settings


def test_build_application_wires_components(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = Settings(
        chunk_size = 500,
        chunk_overlap = 50,
        memory_turns = 3,
    )

    papers_dir = tmp_path / "papers"
    index_dir = tmp_path / "indexes"

    documents = [
        Document(
            page_content = "Research evidence.",
            metadata = {
                "chunk_id": 0,
                "source": "research.pdf",
                "page_number": 1,
            },
        )
    ]

    embeddings = object()
    vector_store = object()
    hybrid_retriever = object()
    reranking_retriever = object()
    model = object()
    query_rewriter = object()
    rag_chain = object()
    sufficiency_chain = object()
    synthesis_chain = object()
    web_search_tool = object()
    checkpointer = object()
    graph = object()

    calls: dict[str, Any] = {}

    def fake_load_corpus(
        papers_dir: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Document]:
        calls["load_corpus"] = {
            "papers_dir": papers_dir,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

        return documents

    monkeypatch.setattr(
        bootstrap,
        "load_corpus",
        fake_load_corpus,
    )

    monkeypatch.setattr(
        bootstrap,
        "build_embeddings",
        lambda settings: embeddings,
    )

    monkeypatch.setattr(
        bootstrap,
        "load_or_build_vector_store",
        lambda **kwargs: vector_store,
    )

    monkeypatch.setattr(
        bootstrap,
        "build_hybrid_retriever",
        lambda **kwargs: hybrid_retriever,
    )

    monkeypatch.setattr(
        bootstrap,
        "build_reranking_retriever",
        lambda **kwargs: reranking_retriever,
    )

    monkeypatch.setattr(
        bootstrap,
        "build_chat_model",
        lambda settings: model,
    )

    monkeypatch.setattr(
        bootstrap,
        "build_query_rewriter",
        lambda model: query_rewriter,
    )

    monkeypatch.setattr(
        bootstrap,
        "build_rag_chain",
        lambda **kwargs: rag_chain,
    )

    monkeypatch.setattr(
        bootstrap,
        "build_sufficiency_chain",
        lambda model: sufficiency_chain,
    )

    monkeypatch.setattr(
        bootstrap,
        "build_synthesis_chain",
        lambda model: synthesis_chain,
    )

    monkeypatch.setattr(
        bootstrap,
        "build_web_search_tool",
        lambda: web_search_tool,
    )

    def fake_build_research_graph(
        **kwargs,
    ):
        calls["graph"] = kwargs
        return graph

    monkeypatch.setattr(
        bootstrap,
        "build_research_graph",
        fake_build_research_graph,
    )

    result = bootstrap.build_application(
        papers_dir = papers_dir,
        index_dir = index_dir,
        settings = settings,
        checkpointer = checkpointer,
    )

    assert result is graph

    assert calls["load_corpus"] == {
        "papers_dir": papers_dir,
        "chunk_size": 500,
        "chunk_overlap": 50,
    }

    assert calls["graph"] == {
        "query_rewriter": query_rewriter,
        "rag_chain": rag_chain,
        "sufficiency_chain": sufficiency_chain,
        "web_search_tool": web_search_tool,
        "synthesis_chain": synthesis_chain,
        "memory_turns": 3,
        "checkpointer": checkpointer,
    }


def test_build_application_uses_in_memory_checkpointer_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = Settings()

    documents = [
        Document(
            page_content = "Evidence.",
            metadata = {
                "chunk_id": 0,
                "source": "research.pdf",
                "page_number": 1,
            },
        )
    ]

    sentinel = object()

    monkeypatch.setattr(
        bootstrap,
        "load_corpus",
        lambda **kwargs: documents,
    )

    monkeypatch.setattr(
        bootstrap,
        "build_embeddings",
        lambda settings: object(),
    )

    monkeypatch.setattr(
        bootstrap,
        "load_or_build_vector_store",
        lambda **kwargs: object(),
    )

    monkeypatch.setattr(
        bootstrap,
        "build_hybrid_retriever",
        lambda **kwargs: object(),
    )

    monkeypatch.setattr(
        bootstrap,
        "build_reranking_retriever",
        lambda **kwargs: object(),
    )

    monkeypatch.setattr(
        bootstrap,
        "build_chat_model",
        lambda settings: object(),
    )

    monkeypatch.setattr(
        bootstrap,
        "build_query_rewriter",
        lambda model: object(),
    )

    monkeypatch.setattr(
        bootstrap,
        "build_rag_chain",
        lambda **kwargs: object(),
    )

    monkeypatch.setattr(
        bootstrap,
        "build_sufficiency_chain",
        lambda model: object(),
    )

    monkeypatch.setattr(
        bootstrap,
        "build_synthesis_chain",
        lambda model: object(),
    )

    monkeypatch.setattr(
        bootstrap,
        "build_web_search_tool",
        lambda: object(),
    )

    checkpointer = object()

    monkeypatch.setattr(
        bootstrap,
        "InMemorySaver",
        lambda: checkpointer,
    )

    def fake_build_research_graph(
        **kwargs,
    ):
        assert kwargs["checkpointer"] is checkpointer
        return sentinel

    monkeypatch.setattr(
        bootstrap,
        "build_research_graph",
        fake_build_research_graph,
    )

    result = bootstrap.build_application(
        papers_dir = tmp_path / "papers",
        index_dir = tmp_path / "indexes",
        settings = settings,
    )

    assert result is sentinel
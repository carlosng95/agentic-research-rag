from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from agentic_research_rag import bootstrap
from agentic_research_rag.config import Settings


def test_build_application_wires_components(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(memory_turns = 3)
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

    def fake_build_embeddings(settings: Settings):
        calls["build_embeddings"] = {
            "settings": settings,
        }

        return embeddings

    def fake_load_index_artifacts(index_dir: Path, embeddings, settings: Settings):
        calls["load_index_artifacts"] = {
            "index_dir": index_dir,
            "embeddings": embeddings,
            "settings": settings,
        }

        return vector_store, documents

    def fake_build_hybrid_retriever(documents, vector_store, settings: Settings):
        calls["build_hybrid_retriever"] = {
            "documents": documents,
            "vector_store": vector_store,
            "settings": settings,
        }

        return hybrid_retriever

    def fake_build_reranking_retriever(base_retriever, settings: Settings):
        calls["build_reranking_retriever"] = {
            "base_retriever": base_retriever,
            "settings": settings,
        }

        return reranking_retriever

    monkeypatch.setattr(bootstrap, "build_embeddings", fake_build_embeddings)
    monkeypatch.setattr(bootstrap, "load_index_artifacts", fake_load_index_artifacts)
    monkeypatch.setattr(bootstrap, "build_hybrid_retriever", fake_build_hybrid_retriever)
    monkeypatch.setattr(bootstrap, "build_reranking_retriever", fake_build_reranking_retriever)
    monkeypatch.setattr(bootstrap, "build_chat_model", lambda settings: model)
    monkeypatch.setattr(bootstrap, "build_query_rewriter", lambda model: query_rewriter)
    monkeypatch.setattr(bootstrap, "build_rag_chain", lambda **kwargs: rag_chain)
    monkeypatch.setattr(bootstrap, "build_sufficiency_chain", lambda model: sufficiency_chain)
    monkeypatch.setattr(bootstrap, "build_synthesis_chain", lambda model: synthesis_chain)
    monkeypatch.setattr(bootstrap, "build_web_search_tool", lambda: web_search_tool)

    def fake_build_research_graph(**kwargs):
        calls["graph"] = kwargs
        return graph

    monkeypatch.setattr(bootstrap, "build_research_graph", fake_build_research_graph)

    result = bootstrap.build_application(
        index_dir = index_dir,
        settings = settings,
        checkpointer = checkpointer,
    )

    assert result is graph

    assert calls["build_embeddings"] == {
        "settings": settings,
    }

    assert calls["load_index_artifacts"] == {
        "index_dir": index_dir,
        "embeddings": embeddings,
        "settings": settings,
    }

    assert calls["build_hybrid_retriever"] == {
        "documents": documents,
        "vector_store": vector_store,
        "settings": settings,
    }

    assert calls["build_reranking_retriever"] == {
        "base_retriever": hybrid_retriever,
        "settings": settings,
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


def test_build_application_uses_in_memory_checkpointer_by_default(monkeypatch, tmp_path: Path) -> None:
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

    embeddings = object()
    vector_store = object()
    checkpointer = object()
    sentinel = object()

    monkeypatch.setattr(bootstrap, "build_embeddings", lambda settings: embeddings)

    monkeypatch.setattr(
        bootstrap,
        "load_index_artifacts",
        lambda **kwargs: (vector_store, documents),
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

    monkeypatch.setattr(bootstrap, "build_chat_model", lambda settings: object())
    monkeypatch.setattr(bootstrap, "build_query_rewriter", lambda model: object())
    monkeypatch.setattr(bootstrap, "build_rag_chain", lambda **kwargs: object())
    monkeypatch.setattr(bootstrap, "build_sufficiency_chain", lambda model: object())
    monkeypatch.setattr(bootstrap, "build_synthesis_chain", lambda model: object())
    monkeypatch.setattr(bootstrap, "build_web_search_tool", lambda: object())

    def fake_build_research_graph(**kwargs):
        assert kwargs["checkpointer"] is checkpointer
        return sentinel

    monkeypatch.setattr(bootstrap, "build_research_graph", fake_build_research_graph)

    result = bootstrap.build_application(
        index_dir = tmp_path / "indexes",
        settings = settings,
        checkpointer = checkpointer,
    )

    assert result is sentinel
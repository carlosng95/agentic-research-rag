from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from .chains.model import build_chat_model
from .chains.query_rewriter import build_query_rewriter
from .chains.rag import build_rag_chain
from .chains.sufficiency import build_sufficiency_chain
from .chains.synthesis import build_synthesis_chain
from .config import Settings
from .graph.research_graph import build_research_graph
from .retrieval.embeddings import build_embeddings
from .retrieval.hybrid import build_hybrid_retriever
from .retrieval.index_manager import load_index_artifacts
from .retrieval.reranker import build_reranking_retriever
from .tools.web_search import build_web_search_tool


def build_application(
    index_dir: str | Path = "data/indexes/faiss",
    settings: Settings | None = None,
    *,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    if settings is None:
        settings = Settings.from_env()

    index_dir = Path(index_dir)

    embeddings = build_embeddings(settings = settings)

    vector_store, documents = load_index_artifacts(
        index_dir = index_dir,
        embeddings = embeddings,
        settings = settings,
    )

    hybrid_retriever = build_hybrid_retriever(
        documents = documents,
        vector_store = vector_store,
        settings = settings,
    )

    retriever = build_reranking_retriever(
        base_retriever = hybrid_retriever,
        settings = settings,
    )

    model = build_chat_model(settings = settings)

    query_rewriter = build_query_rewriter(model = model)

    rag_chain = build_rag_chain(
        retriever = retriever,
        model = model,
    )

    sufficiency_chain = build_sufficiency_chain(model = model)
    synthesis_chain = build_synthesis_chain(model = model)
    web_search_tool = build_web_search_tool()

    return build_research_graph(
        query_rewriter = query_rewriter,
        rag_chain = rag_chain,
        sufficiency_chain = sufficiency_chain,
        web_search_tool = web_search_tool,
        synthesis_chain = synthesis_chain,
        memory_turns = settings.memory_turns,
        checkpointer = checkpointer,
    )
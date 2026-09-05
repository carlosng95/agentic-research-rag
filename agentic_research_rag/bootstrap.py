from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from .chains.model import build_chat_model
from .chains.query_rewriter import build_query_rewriter
from .chains.rag import build_rag_chain
from .chains.sufficiency import build_sufficiency_chain
from .chains.synthesis import build_synthesis_chain
from .config import Settings
from .graph.research_graph import build_research_graph
from .ingestion.documents import load_corpus
from .retrieval.embeddings import build_embeddings
from .retrieval.hybrid import build_hybrid_retriever
from .retrieval.index_manager import (
    load_or_build_vector_store,
)
from .retrieval.reranker import (
    build_reranking_retriever,
)
from .tools.web_search import build_web_search_tool


def build_application(
    papers_dir: str | Path = "papers",
    index_dir: str | Path = "data/indexes/faiss",
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    if settings is None:
        settings = Settings.from_env()

    papers_dir = Path(
        papers_dir
    )

    index_dir = Path(
        index_dir
    )

    documents = load_corpus(
        papers_dir = papers_dir,
        chunk_size = settings.chunk_size,
        chunk_overlap = settings.chunk_overlap,
    )

    embeddings = build_embeddings(
        settings = settings
    )

    vector_store = load_or_build_vector_store(
        documents = documents,
        embeddings = embeddings,
        settings = settings,
        index_dir = index_dir,
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

    model = build_chat_model(
        settings = settings
    )

    query_rewriter = build_query_rewriter(
        model = model
    )

    rag_chain = build_rag_chain(
        retriever = retriever,
        model = model,
    )

    sufficiency_chain = build_sufficiency_chain(
        model = model
    )

    synthesis_chain = build_synthesis_chain(
        model = model
    )

    web_search_tool = build_web_search_tool()

    if checkpointer is None:
        checkpointer = InMemorySaver()

    return build_research_graph(
        query_rewriter = query_rewriter,
        rag_chain = rag_chain,
        sufficiency_chain = sufficiency_chain,
        web_search_tool = web_search_tool,
        synthesis_chain = synthesis_chain,
        memory_turns = settings.memory_turns,
        checkpointer = checkpointer,
    )
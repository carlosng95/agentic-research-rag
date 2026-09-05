from pathlib import Path

from .agents.research_agent import ResearchAgent
from .config import Settings
from .evaluation.sufficiency import SufficiencyEvaluator
from .indexing.bm25_index import BM25Index
from .indexing.semantic_index import SemanticIndex
from .ingestion.corpus import load_corpus
from .memory import ConversationMemory
from .providers.embeddings import LocalEmbeddingProvider
from .providers.llm import OpenAILLMProvider
from .providers.websearch import OpenAIWebSearchProvider
from .query_rewriter import QueryRewriter
from .rag_pipeline import RAGPipeline
from .reranking.cross_encoder import CrossEncoderReranker
from .retrieval_pipeline import RetrievalPipeline
from .retriever import Retriever
from .synthesis.evidence import EvidenceSynthesizer
from .tokenizer import RegexTokenizer
from .tools.paper_rag import PaperRAGTool
from .tools.websearch import WebSearchTool
from .indexing.semantic_index_manager import load_or_build_semantic_index


def build_research_agent(
    papers_dir: str | Path,
    semantic_index_path: str | Path,
) -> ResearchAgent:
    papers_dir = Path(papers_dir)
    semantic_index_path = Path(semantic_index_path)

    settings = Settings.from_env()

    chunks = load_corpus(
        papers_dir = papers_dir,
        chunk_size = settings.chunk_size,
        overlap = settings.chunk_overlap,
    )

    embedding_provider = LocalEmbeddingProvider()

    semantic_index = load_or_build_semantic_index(
        path = semantic_index_path,
        chunks = chunks,
        embedding_provider = embedding_provider,
    )

    tokenizer = RegexTokenizer()

    bm25_index = BM25Index(
        chunks = chunks,
        tokenizer = tokenizer,
    )

    retriever = Retriever(
        semantic_index = semantic_index,
        bm25_index = bm25_index,
        rrf_k = settings.rrf_k,
    )

    reranker = CrossEncoderReranker()

    retrieval_pipeline = RetrievalPipeline(
        retriever = retriever,
        reranker = reranker,
        candidate_k = settings.candidate_k,
        rerank_k = settings.rerank_k,
        final_k = settings.final_k,
    )

    llm = OpenAILLMProvider()

    rag_pipeline = RAGPipeline(
        retrieval_pipeline = retrieval_pipeline,
        llm = llm,
    )

    paper_tool = PaperRAGTool(
        rag_pipeline = rag_pipeline,
    )

    web_search_provider = OpenAIWebSearchProvider()

    web_tool = WebSearchTool(
        web_search_provider = web_search_provider,
    )

    sufficiency_evaluator = SufficiencyEvaluator(
        llm = llm,
    )

    synthesizer = EvidenceSynthesizer(
        llm = llm,
    )

    memory = ConversationMemory(
        max_turns = settings.memory_turns,
    )

    query_rewriter = QueryRewriter(
        llm = llm,
    )

    return ResearchAgent(
        paper_tool = paper_tool,
        web_tool = web_tool,
        sufficiency_evaluator = sufficiency_evaluator,
        synthesizer = synthesizer,
        memory = memory,
        query_rewriter = query_rewriter,
    )
    

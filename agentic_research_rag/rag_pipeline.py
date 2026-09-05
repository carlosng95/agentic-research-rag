from .context_builder import build_context
from .prompt_builder import build_rag_prompt
from .providers.llm import LLMProvider
from .retrieval_pipeline import RetrievalPipeline
from .types import Chunk, ResearchResponse, Source
from .citation_validator import validate_citations

class RAGPipeline:
    """
    Coordinate retrieval, context construction and answer generation.
    """

    def __init__(self, retrieval_pipeline: RetrievalPipeline, llm: LLMProvider) -> None:
        self._retrieval_pipeline = retrieval_pipeline
        self._llm = llm

    @staticmethod
    def _chunks_to_sources(chunks: list[Chunk]) -> list[Source]:
        sources: list[Source] = []

        for chunk in chunks:
            source = Source(
                type = "paper",
                ref = chunk.document_name,
                locator = f"page {chunk.page_number}",
                snippet = chunk.text,
            )

            sources.append(source)

        return sources

    def answer(
        self,
        query: str
    ) -> ResearchResponse:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        chunks = self._retrieval_pipeline.search(
            query = query,
        )

        if not chunks:
            return ResearchResponse(
                answer = "No relevant information was found in the available papers.",
                sources = [],
                reasoning_steps = [],
                confidence = 0.0,
                flags = ["no_retrieval_results"],
            )

        context = build_context(chunks)

        prompt = build_rag_prompt(
            query = query,
            context = context,
        )

        answer = self._llm.generate(
            prompt = prompt,
            
        )
        flags = validate_citations(
            answer = answer,
            source_count = len(chunks),
        )

        sources = self._chunks_to_sources(chunks)

        return ResearchResponse(
            answer = answer,
            sources = sources,
            reasoning_steps = [
                f"Retrieved and reranked {len(chunks)} document chunks.",
                "Generated the answer using the retrieved paper context.",
            ],
            confidence = 0.0,
            flags = flags,
        )
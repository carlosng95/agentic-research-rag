from ..citation_validator import validate_citations
from ..providers.llm import LLMProvider
from ..types import ResearchResponse, Source


class EvidenceSynthesizer:
    """
    Combine paper and web evidence into a single grounded response.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    @staticmethod
    def _merge_sources(
        paper_response: ResearchResponse,
        web_response: ResearchResponse,
    ) -> list[Source]:
        sources: list[Source] = []
        seen: set[tuple[str, str, str | None]] = set()

        for source in paper_response.sources + web_response.sources:
            key = (
                source.type,
                source.ref,
                source.locator,
            )

            if key in seen:
                continue

            seen.add(key)
            sources.append(source)

        return sources

    @staticmethod
    def _build_source_context(sources: list[Source]) -> str:
        sections: list[str] = []

        for index, source in enumerate(sources, start = 1):
            section = (
                f"[SOURCE {index}]\n"
                f"Type: {source.type}\n"
                f"Reference: {source.ref}\n"
                f"Location: {source.locator or 'N/A'}"
            )

            if source.snippet:
                section += f"\n\nEvidence:\n{source.snippet}"

            sections.append(section)

        return "\n\n---\n\n".join(sections)

    def synthesize(
        self,
        query: str,
        paper_response: ResearchResponse,
        web_response: ResearchResponse,
    ) -> ResearchResponse:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        sources = self._merge_sources(
            paper_response = paper_response,
            web_response = web_response,
        )

        source_context = self._build_source_context(sources)

        prompt = (
            "You are a research assistant combining evidence from scientific "
            "papers and web research.\n\n"
            "Answer the user's question using the available evidence.\n\n"
            "Instructions:\n"
            "- Prefer scientific paper evidence when it directly answers the question.\n"
            "- Use web evidence to complement information missing from the papers.\n"
            "- Do not invent facts that are not supported by the provided evidence.\n"
            "- Cite claims using the unified [SOURCE N] identifiers below.\n"
            "- Do not reuse citation numbers from the intermediate answers.\n"
            "- If the evidence is still insufficient, say so clearly.\n\n"
            f"QUESTION:\n{query}\n\n"
            f"PAPER RESPONSE:\n{paper_response.answer}\n\n"
            f"WEB RESPONSE:\n{web_response.answer}\n\n"
            f"UNIFIED SOURCES:\n{source_context}\n\n"
            "FINAL ANSWER:"
        )

        answer = self._llm.generate(
            prompt = prompt,
        )

        flags = validate_citations(
            answer = answer,
            source_count = len(sources),
        )

        return ResearchResponse(
            answer = answer,
            sources = sources,
            reasoning_steps = [
                "Paper evidence was evaluated as insufficient.",
                "Searched the web for complementary evidence.",
                "Synthesized paper and web evidence into a final answer.",
                "Validated citations against the combined source set.",
            ],
            confidence = 0.0,
            flags = flags,
        )
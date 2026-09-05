from ..providers.llm import LLMProvider
from ..types import ResearchResponse


class SufficiencyEvaluator:
    """
    Evaluate whether retrieved paper evidence is sufficient
    to answer a user query.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    @staticmethod
    def _build_evidence(response: ResearchResponse) -> str:
        sections: list[str] = []

        for index, source in enumerate(response.sources, start = 1):
            section = (
                f"[SOURCE {index}]\n"
                f"Document: {source.ref}\n"
                f"Location: {source.locator}\n\n"
                f"{source.snippet or ''}"
            )

            sections.append(section)

        return "\n\n---\n\n".join(sections)

    def is_sufficient(self, query: str, response: ResearchResponse) -> bool:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not response.sources:
            return False

        evidence = self._build_evidence(response)

        prompt = (
            "You are evaluating whether retrieved scientific evidence is sufficient "
            "to answer a user's question accurately.\n\n"
            "Determine whether the provided evidence directly contains enough "
            "information to answer the question without relying on external knowledge.\n\n"
            "Use these criteria:\n"
            "- The evidence must be relevant to the question.\n"
            "- The evidence must contain enough information to support a meaningful answer.\n"
            "- Do not assume facts that are not present in the evidence.\n"
            "- If important information is missing, consider the evidence insufficient.\n\n"
            f"QUESTION:\n{query}\n\n"
            f"EVIDENCE:\n{evidence}\n\n"
            "Return only one word:\n"
            "SUFFICIENT\n"
            "or\n"
            "INSUFFICIENT"
        )

        decision = self._llm.generate(prompt = prompt)
        decision = decision.strip().upper().splitlines()[0].strip(" .,:;")

        if decision == "SUFFICIENT":
            return True

        if decision == "INSUFFICIENT":
            return False

        raise ValueError(
            f"Unexpected sufficiency decision: '{decision}'."
        )
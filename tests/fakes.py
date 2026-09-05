from agentic_research_rag.providers.llm import LLMProvider


class FakeLLMProvider(LLMProvider):
    """
    Deterministic LLM provider used in tests.
    """

    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "fake-llm"

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)

        return self._response
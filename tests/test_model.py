import pytest

from agentic_research_rag.config import Settings
from agentic_research_rag.chains import model as model_module


class FakeChatModel:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_retries: int,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries


def test_build_chat_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_chat_openai(
        model: str,
        temperature: float,
        max_retries: int,
    ) -> FakeChatModel:
        captured["model"] = model
        captured["temperature"] = temperature
        captured["max_retries"] = max_retries

        return FakeChatModel(
            model = model,
            temperature = temperature,
            max_retries = max_retries,
        )

    monkeypatch.setattr(
        model_module,
        "ChatOpenAI",
        fake_chat_openai,
    )

    settings = Settings(
        llm_model = "test-model",
        llm_temperature = 0.2,
    )

    model = model_module.build_chat_model(
        settings = settings
    )

    assert isinstance(
        model,
        FakeChatModel,
    )

    assert captured == {
        "model": "test-model",
        "temperature": 0.2,
        "max_retries": 2,
    }
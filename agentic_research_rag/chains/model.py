from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..config import Settings
from ..observability.llm import LLMMetricsCallback


def build_chat_model(settings: Settings) -> BaseChatModel:
    return ChatOpenAI(
        model = settings.llm_model,
        temperature = settings.llm_temperature,
        max_retries = 2,
        callbacks = [
            LLMMetricsCallback(
                model_name = settings.llm_model
            )
        ],
    )
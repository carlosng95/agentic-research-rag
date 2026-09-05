from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from ..config import Settings


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_backend == "local":
        return HuggingFaceEmbeddings(
            model_name = settings.local_embedding_model,
        )

    if settings.embedding_backend == "openai":
        return OpenAIEmbeddings(
            model = settings.openai_embedding_model,
        )

    raise ValueError(
        f"Unsupported embedding backend: {settings.embedding_backend}"
    )
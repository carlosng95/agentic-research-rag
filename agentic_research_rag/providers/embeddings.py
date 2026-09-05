from abc import ABC, abstractmethod
from ..config import get_required_env

class EmbeddingUnavailableError(RuntimeError):
    """
    Raised when embeddings cannot be generated.
    """

    pass


class EmbeddingProvider(ABC):
    """
    Interface implemented by every embedding provider.
    """

    available: bool

    @property
    @abstractmethod
    def model_id(self) -> str:
        """
        Unique identifier of the embedding model.
        """

        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate one embedding vector for each input text.

        The returned vectors must preserve the same order
        as the input texts.
        """

        raise NotImplementedError


class NullEmbeddingProvider(EmbeddingProvider):
    """
    Provider used when embeddings are not configured.
    """

    available = False

    @property
    def model_id(self) -> str:
        return "none"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingUnavailableError(
            "No embedding provider configured."
        )


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider based on sentence-transformers.

    Embeddings are generated locally without calling
    an external API.
    """

    available = True

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model_name = model_name or get_required_env('LOCAL_EMBEDDING_MODEL')
            self._model = SentenceTransformer(self._model_name)

        except Exception as error:
            raise EmbeddingUnavailableError(
                f"Could not load local embedding model: {error}"
            ) from error

    @property
    def model_id(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings = True,
            )

        except Exception as error:
            raise EmbeddingUnavailableError(
                f"Local embedding generation failed: {error}"
            ) from error

        return embeddings.tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider backed by the OpenAI API.
    """

    available = True

    def __init__(self, api_key: str, model: str | None = None) -> None:
        try:
            from openai import OpenAI
            self._model = model or get_required_env("OPENAI_EMBEDDING_MODEL")
            self._client = OpenAI(api_key = api_key)

        except Exception as error:
            raise EmbeddingUnavailableError(
                f"Could not initialize OpenAI client: {error}"
            ) from error

        self._model = model

    @property
    def model_id(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = self._client.embeddings.create(
                model = self._model,
                input = texts,
            )

        except Exception as error:
            raise EmbeddingUnavailableError(
                f"Embedding generation failed: {error}"
            ) from error

        return [item.embedding for item in response.data]
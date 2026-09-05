import os

from dataclasses import dataclass


def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise ValueError(f"Environment variable '{name}' is not configured")

    return value


def get_int_env(name: str, default: int | None = None) -> int:
    value = os.getenv(name)

    if value is None:
        if default is not None:
            return default

        raise ValueError(f"Environment variable '{name}' is not configured")

    try:
        return int(value)

    except ValueError as error:
        raise ValueError(
            f"Environment variable '{name}' must be an integer"
        ) from error


@dataclass(frozen = True)
class Settings:
    chunk_size: int
    chunk_overlap: int
    rrf_k: int
    candidate_k: int
    rerank_k: int
    final_k: int
    memory_turns: int

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            chunk_size = get_int_env("CHUNK_SIZE", default = 1200),
            chunk_overlap = get_int_env("CHUNK_OVERLAP", default = 200),
            rrf_k = get_int_env("RRF_K", default = 60),
            candidate_k = get_int_env("RETRIEVAL_CANDIDATE_K", default = 30),
            rerank_k = get_int_env("RERANK_K", default = 20),
            final_k = get_int_env("FINAL_K", default = 5),
            memory_turns = get_int_env("MEMORY_TURNS", default = 5),
        )

        settings.validate()

        return settings

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        if self.rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")

        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")

        if self.rerank_k <= 0:
            raise ValueError("rerank_k must be greater than 0")

        if self.final_k <= 0:
            raise ValueError("final_k must be greater than 0")

        if self.rerank_k > self.candidate_k:
            raise ValueError("rerank_k cannot be greater than candidate_k")

        if self.final_k > self.rerank_k:
            raise ValueError("final_k cannot be greater than rerank_k")

        if self.memory_turns <= 0:
            raise ValueError("memory_turns must be greater than 0")
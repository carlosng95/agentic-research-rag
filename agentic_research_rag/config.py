from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen = True)
class Settings:
    chunk_size: int = 1200
    chunk_overlap: int = 200

    index_dir: str = "data/indexes/faiss"
    artifact_source: str = "local"

    aws_region: str = "us-east-1"
    artifact_bucket: str = ""
    artifact_prefix: str = "agentic-research-rag"

    embedding_backend: str = "local"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    rrf_k: int = 60
    semantic_weight: float = 0.5
    bm25_weight: float = 0.5

    candidate_k: int = 30
    rerank_k: int = 20
    final_k: int = 5

    memory_turns: int = 5

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        if not self.index_dir.strip():
            raise ValueError("index_dir cannot be empty.")

        if self.artifact_source not in {"local", "s3"}:
            raise ValueError("artifact_source must be either 'local' or 's3'.")

        if not self.aws_region.strip():
            raise ValueError("aws_region cannot be empty.")

        if not self.artifact_prefix.strip():
            raise ValueError("artifact_prefix cannot be empty.")

        if self.embedding_backend not in {"local", "openai"}:
            raise ValueError(
                "embedding_backend must be either 'local' or 'openai'."
            )

        if not self.local_embedding_model.strip():
            raise ValueError("local_embedding_model cannot be empty.")

        if not self.openai_embedding_model.strip():
            raise ValueError("openai_embedding_model cannot be empty.")

        if not self.cross_encoder_model.strip():
            raise ValueError("cross_encoder_model cannot be empty.")

        if not self.llm_model.strip():
            raise ValueError("llm_model cannot be empty.")

        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError(
                "llm_temperature must be between 0.0 and 2.0."
            )

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0.")

        if self.semantic_weight < 0:
            raise ValueError("semantic_weight cannot be negative.")

        if self.bm25_weight < 0:
            raise ValueError("bm25_weight cannot be negative.")

        if self.semantic_weight + self.bm25_weight <= 0:
            raise ValueError(
                "At least one retrieval weight must be greater than 0."
            )

        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0.")

        if self.rerank_k <= 0:
            raise ValueError("rerank_k must be greater than 0.")

        if self.rerank_k > self.candidate_k * 2:
            raise ValueError(
                "rerank_k cannot exceed twice candidate_k."
            )

        if self.final_k <= 0:
            raise ValueError("final_k must be greater than 0.")

        if self.final_k > self.rerank_k:
            raise ValueError("final_k cannot exceed rerank_k.")

        if self.memory_turns < 0:
            raise ValueError("memory_turns cannot be negative.")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        return cls(
            chunk_size = int(
                os.getenv("CHUNK_SIZE", "1200")
            ),
            chunk_overlap = int(
                os.getenv("CHUNK_OVERLAP", "200")
            ),

            index_dir = os.getenv(
                "INDEX_DIR",
                "data/indexes/faiss",
            ),
            artifact_source = os.getenv(
                "ARTIFACT_SOURCE",
                "local",
            ),

            aws_region = os.getenv(
                "AWS_REGION",
                "us-east-1",
            ),
            artifact_bucket = os.getenv(
                "ARTIFACT_BUCKET",
                "",
            ),
            artifact_prefix = os.getenv(
                "ARTIFACT_PREFIX",
                "agentic-research-rag",
            ),

            embedding_backend = os.getenv(
                "EMBEDDING_BACKEND",
                "local",
            ),
            local_embedding_model = os.getenv(
                "LOCAL_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            openai_embedding_model = os.getenv(
                "OPENAI_EMBEDDING_MODEL",
                "text-embedding-3-small",
            ),

            cross_encoder_model = os.getenv(
                "CROSS_ENCODER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L6-v2",
            ),

            llm_model = os.getenv(
                "LLM_MODEL",
                "gpt-4o-mini",
            ),
            llm_temperature = float(
                os.getenv("LLM_TEMPERATURE", "0.0")
            ),

            rrf_k = int(
                os.getenv("RRF_K", "60")
            ),
            semantic_weight = float(
                os.getenv("SEMANTIC_WEIGHT", "0.5")
            ),
            bm25_weight = float(
                os.getenv("BM25_WEIGHT", "0.5")
            ),

            candidate_k = int(
                os.getenv("RETRIEVAL_CANDIDATE_K", "30")
            ),
            rerank_k = int(
                os.getenv("RERANK_K", "20")
            ),
            final_k = int(
                os.getenv("FINAL_K", "5")
            ),

            memory_turns = int(
                os.getenv("MEMORY_TURNS", "5")
            ),
        )
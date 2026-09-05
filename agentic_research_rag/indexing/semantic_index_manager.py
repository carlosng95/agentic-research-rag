from pathlib import Path

from ..providers.embeddings import EmbeddingProvider
from ..types import Chunk
from .semantic_index import SemanticIndex, SemanticIndexMismatchError


def load_or_build_semantic_index(
    path: str | Path,
    chunks: list[Chunk],
    embedding_provider: EmbeddingProvider,
) -> SemanticIndex:
    path = Path(path)

    if path.exists():
        try:
            return SemanticIndex.load(
                path = path,
                chunks = chunks,
                embedding_provider = embedding_provider,
            )

        except SemanticIndexMismatchError:
            pass

    semantic_index = SemanticIndex.build(
        chunks = chunks,
        embedding_provider = embedding_provider,
    )

    semantic_index.save(path)

    return semantic_index
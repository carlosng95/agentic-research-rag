import hashlib
import json
from pathlib import Path
import shutil

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ..config import Settings


_METADATA_FILENAME = "metadata.json"
_INDEX_NAME = "index"


def _corpus_fingerprint(
    documents: list[Document],
) -> str:
    hasher = hashlib.sha256()

    for document in documents:
        chunk_id = document.metadata.get("chunk_id", "")
        source = document.metadata.get("source", "")
        page_number = document.metadata.get("page_number", "")

        payload = (
            f"{chunk_id}\n"
            f"{source}\n"
            f"{page_number}\n"
            f"{document.page_content}\n"
        )

        hasher.update(
            payload.encode("utf-8")
        )

    return hasher.hexdigest()


def _embedding_model_id(
    settings: Settings,
) -> str:
    if settings.embedding_backend == "local":
        return (
            f"local:"
            f"{settings.local_embedding_model}"
        )

    if settings.embedding_backend == "openai":
        return (
            f"openai:"
            f"{settings.openai_embedding_model}"
        )

    raise ValueError(
        f"Unsupported embedding backend: "
        f"{settings.embedding_backend}"
    )


def _build_metadata(
    documents: list[Document],
    settings: Settings,
) -> dict[str, str | int]:
    return {
        "corpus_fingerprint": _corpus_fingerprint(
            documents = documents
        ),
        "embedding_model_id": _embedding_model_id(
            settings = settings
        ),
        "document_count": len(documents),
    }


def _metadata_path(
    index_dir: Path,
) -> Path:
    return index_dir / _METADATA_FILENAME


def _load_metadata(
    index_dir: Path,
) -> dict[str, str | int] | None:
    path = _metadata_path(
        index_dir = index_dir
    )

    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding = "utf-8"
            )
        )
    except (
        json.JSONDecodeError,
        OSError,
    ):
        return None


def _save_metadata(
    index_dir: Path,
    metadata: dict[str, str | int],
) -> None:
    path = _metadata_path(
        index_dir = index_dir
    )

    path.write_text(
        json.dumps(
            metadata,
            indent = 2,
            sort_keys = True,
        ),
        encoding = "utf-8",
    )


def _is_index_valid(
    index_dir: Path,
    documents: list[Document],
    settings: Settings,
) -> bool:
    stored_metadata = _load_metadata(
        index_dir = index_dir
    )

    if stored_metadata is None:
        return False

    current_metadata = _build_metadata(
        documents = documents,
        settings = settings,
    )

    return stored_metadata == current_metadata


def build_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
) -> FAISS:
    if not documents:
        raise ValueError(
            "Cannot build a vector store from an empty corpus."
        )

    return FAISS.from_documents(
        documents = documents,
        embedding = embeddings,
        normalize_L2 = True,
        distance_strategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
    )


def save_vector_store(
    vector_store: FAISS,
    index_dir: str | Path,
    documents: list[Document],
    settings: Settings,
) -> None:
    index_dir = Path(index_dir)

    index_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    vector_store.save_local(
        folder_path = str(index_dir),
        index_name = _INDEX_NAME,
    )

    metadata = _build_metadata(
        documents = documents,
        settings = settings,
    )

    _save_metadata(
        index_dir = index_dir,
        metadata = metadata,
    )


def load_vector_store(
    index_dir: str | Path,
    embeddings: Embeddings,
) -> FAISS:
    index_dir = Path(index_dir)

    return FAISS.load_local(
        folder_path = str(index_dir),
        embeddings = embeddings,
        index_name = _INDEX_NAME,
        allow_dangerous_deserialization = True,
    )


def load_or_build_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    settings: Settings,
    index_dir: str | Path,
) -> FAISS:
    index_dir = Path(index_dir)

    if _is_index_valid(
        index_dir = index_dir,
        documents = documents,
        settings = settings,
    ):
        try:
            return load_vector_store(
                index_dir = index_dir,
                embeddings = embeddings,
            )
        except Exception:
            pass

    if index_dir.exists():
        shutil.rmtree(index_dir)

    vector_store = build_vector_store(
        documents = documents,
        embeddings = embeddings,
    )

    save_vector_store(
        vector_store = vector_store,
        index_dir = index_dir,
        documents = documents,
        settings = settings,
    )

    return vector_store
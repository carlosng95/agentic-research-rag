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
_DOCUMENTS_FILENAME = "documents.jsonl"
_INDEX_NAME = "index"
_SCHEMA_VERSION = 1


def _corpus_fingerprint(documents: list[Document]) -> str:
    hasher = hashlib.sha256()

    for document in documents:
        chunk_id = document.metadata.get("chunk_id", "")
        source = document.metadata.get("source", "")
        page_number = document.metadata.get("page_number", "")

        payload = f"{chunk_id}\n{source}\n{page_number}\n{document.page_content}\n"
        hasher.update(payload.encode("utf-8"))

    return hasher.hexdigest()


def _embedding_model_id(settings: Settings) -> str:
    if settings.embedding_backend == "local":
        return f"local:{settings.local_embedding_model}"

    if settings.embedding_backend == "openai":
        return f"openai:{settings.openai_embedding_model}"

    raise ValueError(f"Unsupported embedding backend: {settings.embedding_backend}")


def _build_metadata(documents: list[Document], settings: Settings) -> dict[str, str | int]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "corpus_fingerprint": _corpus_fingerprint(documents = documents),
        "embedding_model_id": _embedding_model_id(settings = settings),
        "document_count": len(documents),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }


def _metadata_path(index_dir: Path) -> Path:
    return index_dir / _METADATA_FILENAME


def _documents_path(index_dir: Path) -> Path:
    return index_dir / _DOCUMENTS_FILENAME


def _load_metadata(index_dir: Path) -> dict[str, str | int] | None:
    path = _metadata_path(index_dir = index_dir)

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding = "utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_metadata(index_dir: Path, metadata: dict[str, str | int]) -> None:
    path = _metadata_path(index_dir = index_dir)

    path.write_text(
        json.dumps(metadata, indent = 2, sort_keys = True),
        encoding = "utf-8",
    )


def save_documents(documents: list[Document], index_dir: str | Path) -> None:
    index_dir = Path(index_dir)
    path = _documents_path(index_dir = index_dir)

    with path.open("w", encoding = "utf-8") as file:
        for document in documents:
            payload = {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }

            file.write(json.dumps(payload, ensure_ascii = False))
            file.write("\n")


def load_documents(index_dir: str | Path) -> list[Document]:
    index_dir = Path(index_dir)
    path = _documents_path(index_dir = index_dir)

    if not path.exists():
        raise FileNotFoundError(f"Index documents not found: {path}")

    documents: list[Document] = []

    with path.open("r", encoding = "utf-8") as file:
        for line_number, line in enumerate(file, start = 1):
            line = line.strip()

            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}.") from exc

            documents.append(
                Document(
                    page_content = payload["page_content"],
                    metadata = payload["metadata"],
                )
            )

    if not documents:
        raise ValueError(f"No documents found in index artifact: {path}")

    return documents


def _is_index_valid(index_dir: Path, documents: list[Document], settings: Settings) -> bool:
    stored_metadata = _load_metadata(index_dir = index_dir)

    if stored_metadata is None:
        return False

    current_metadata = _build_metadata(documents = documents, settings = settings)

    return stored_metadata == current_metadata


def build_vector_store(documents: list[Document], embeddings: Embeddings) -> FAISS:
    if not documents:
        raise ValueError("Cannot build a vector store from an empty corpus.")

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
    index_dir.mkdir(parents = True, exist_ok = True)

    vector_store.save_local(
        folder_path = str(index_dir),
        index_name = _INDEX_NAME,
    )

    save_documents(
        documents = documents,
        index_dir = index_dir,
    )

    metadata = _build_metadata(
        documents = documents,
        settings = settings,
    )

    _save_metadata(
        index_dir = index_dir,
        metadata = metadata,
    )


def load_vector_store(index_dir: str | Path, embeddings: Embeddings) -> FAISS:
    index_dir = Path(index_dir)

    return FAISS.load_local(
        folder_path = str(index_dir),
        embeddings = embeddings,
        index_name = _INDEX_NAME,
        allow_dangerous_deserialization = True,
    )


def load_index_artifacts(
    index_dir: str | Path,
    embeddings: Embeddings,
    settings: Settings,
) -> tuple[FAISS, list[Document]]:
    index_dir = Path(index_dir)

    required_files = [
        index_dir / f"{_INDEX_NAME}.faiss",
        index_dir / f"{_INDEX_NAME}.pkl",
        _metadata_path(index_dir = index_dir),
        _documents_path(index_dir = index_dir),
    ]

    missing_files = [path.name for path in required_files if not path.exists()]

    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"Incomplete index artifact in {index_dir}. Missing: {missing}")

    metadata = _load_metadata(index_dir = index_dir)

    if metadata is None:
        raise ValueError(f"Invalid index metadata: {_metadata_path(index_dir = index_dir)}")

    if metadata.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported index schema version: {metadata.get('schema_version')}. "
            f"Expected: {_SCHEMA_VERSION}."
        )

    expected_embedding_model = _embedding_model_id(settings = settings)
    stored_embedding_model = metadata.get("embedding_model_id")

    if stored_embedding_model != expected_embedding_model:
        raise ValueError(
            f"Embedding model mismatch. Index uses '{stored_embedding_model}', "
            f"but serving is configured for '{expected_embedding_model}'."
        )

    documents = load_documents(index_dir = index_dir)

    if metadata.get("document_count") != len(documents):
        raise ValueError(
            f"Index document count mismatch. Metadata expects "
            f"{metadata.get('document_count')}, found {len(documents)}."
        )

    stored_fingerprint = metadata.get("corpus_fingerprint")
    current_fingerprint = _corpus_fingerprint(documents = documents)

    if stored_fingerprint != current_fingerprint:
        raise ValueError("Index corpus fingerprint does not match stored documents.")

    vector_store = load_vector_store(
        index_dir = index_dir,
        embeddings = embeddings,
    )

    return vector_store, documents


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
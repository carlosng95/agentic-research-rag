import argparse
from pathlib import Path
import shutil

from ..config import Settings
from ..ingestion.corpus_source import materialize_corpus_source
from ..ingestion.documents import load_corpus
from ..retrieval.embeddings import build_embeddings
from ..retrieval.index_manager import build_vector_store, save_vector_store


def build_index(
    papers_dir: str | Path = "papers",
    index_dir: str | Path = "data/indexes/faiss",
    settings: Settings | None = None,
    force: bool = False,
) -> None:
    if settings is None:
        settings = Settings.from_env()

    papers_dir = Path(papers_dir)
    index_dir = Path(index_dir)

    if index_dir.exists() and any(index_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Index directory is not empty: {index_dir}. "
                "Use --force to replace the existing index."
            )

        shutil.rmtree(
            index_dir
        )

    with materialize_corpus_source(
        settings = settings,
        local_papers_dir = papers_dir,
    ) as corpus_dir:
        documents = load_corpus(
            papers_dir = corpus_dir,
            chunk_size = settings.chunk_size,
            chunk_overlap = settings.chunk_overlap,
        )

    embeddings = build_embeddings(
        settings = settings
    )

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

    print("Index built successfully.")
    print(f"Documents: {len(documents)}")
    print(f"Index directory: {index_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Build retrieval index artifacts."
    )

    parser.add_argument(
        "--papers-dir",
        default = "papers",
        help = (
            "Local source PDF directory. "
            "Used when CORPUS_SOURCE=local."
        ),
    )

    parser.add_argument(
        "--index-dir",
        default = "data/indexes/faiss",
        help = "Directory where index artifacts will be written.",
    )

    parser.add_argument(
        "--force",
        action = "store_true",
        help = "Replace an existing index directory.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_index(
        papers_dir = args.papers_dir,
        index_dir = args.index_dir,
        force = args.force,
    )


if __name__ == "__main__":
    main()
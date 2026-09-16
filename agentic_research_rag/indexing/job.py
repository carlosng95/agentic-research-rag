from pathlib import Path
from tempfile import TemporaryDirectory

from ..artifacts.publish import upload_artifact
from ..config import Settings
from .build import build_index


def run_indexing_job(
    settings: Settings | None = None,
) -> str:
    if settings is None:
        settings = Settings.from_env()

    with TemporaryDirectory(
        prefix = "agentic-rag-index-",
    ) as temporary_directory:
        index_dir = Path(
            temporary_directory
        )

        build_index(
            index_dir = index_dir,
            settings = settings,
            force = True,
        )

        version = upload_artifact(
            index_dir = index_dir,
            settings = settings,
        )

    return version


def main() -> None:
    version = run_indexing_job()

    print("Indexing job completed successfully.")
    print(f"Version: {version}")
    print("The new version has not been promoted.")


if __name__ == "__main__":
    main()
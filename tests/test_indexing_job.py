from pathlib import Path

from agentic_research_rag.config import Settings
from agentic_research_rag.indexing import job


def test_run_indexing_job(
    monkeypatch,
) -> None:
    settings = Settings()

    calls: list[str] = []

    def fake_build_index(
        index_dir: str | Path,
        settings: Settings,
        force: bool,
    ) -> None:
        calls.append("build")

        assert force is True

        index_dir = Path(
            index_dir
        )

        assert index_dir.exists()

    def fake_upload_artifact(
        index_dir: str | Path,
        settings: Settings,
    ) -> str:
        calls.append("upload")

        index_dir = Path(
            index_dir
        )

        assert index_dir.exists()

        return "20260915T230000Z-test1234"

    monkeypatch.setattr(
        job,
        "build_index",
        fake_build_index,
    )

    monkeypatch.setattr(
        job,
        "upload_artifact",
        fake_upload_artifact,
    )

    version = job.run_indexing_job(
        settings = settings,
    )

    assert version == "20260915T230000Z-test1234"
    assert calls == [
        "build",
        "upload",
    ]
import pytest

from agentic_research_rag.config import Settings


def test_aws_settings_have_safe_defaults() -> None:
    settings = Settings()

    assert settings.aws_region == "us-east-1"
    assert settings.artifact_bucket == ""
    assert settings.artifact_prefix == "agentic-research-rag"


def test_settings_from_env_loads_aws_configuration(monkeypatch) -> None:
    monkeypatch.setenv("AWS_REGION", "sa-east-1")
    monkeypatch.setenv("ARTIFACT_BUCKET", "test-artifact-bucket")
    monkeypatch.setenv("ARTIFACT_PREFIX", "test-prefix")

    settings = Settings.from_env()

    assert settings.aws_region == "sa-east-1"
    assert settings.artifact_bucket == "test-artifact-bucket"
    assert settings.artifact_prefix == "test-prefix"


def test_settings_rejects_empty_aws_region() -> None:
    with pytest.raises(ValueError, match = "aws_region"):
        Settings(aws_region = "   ")


def test_settings_rejects_empty_artifact_prefix() -> None:
    with pytest.raises(ValueError, match = "artifact_prefix"):
        Settings(artifact_prefix = "   ")
from langgraph.checkpoint.memory import InMemorySaver

from agentic_research_rag import checkpointing
from agentic_research_rag.checkpointing import open_checkpointer
from agentic_research_rag.config import Settings


def test_open_checkpointer_memory() -> None:
    settings = Settings(
        checkpoint_backend = "memory",
    )

    with open_checkpointer(settings) as checkpointer:
        assert isinstance(checkpointer, InMemorySaver)


def test_open_checkpointer_postgres(monkeypatch) -> None:
    class FakeCheckpointer:
        def __init__(self) -> None:
            self.setup_called = False

        def setup(self) -> None:
            self.setup_called = True

    class FakeContext:
        def __init__(self, checkpointer) -> None:
            self.checkpointer = checkpointer
            self.enter_called = False
            self.exit_called = False

        def __enter__(self):
            self.enter_called = True
            return self.checkpointer

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            self.exit_called = True

    fake_checkpointer = FakeCheckpointer()
    fake_context = FakeContext(fake_checkpointer)

    monkeypatch.setattr(
        checkpointing.PostgresSaver,
        "from_conn_string",
        lambda connection_string: fake_context,
    )

    settings = Settings(
        checkpoint_backend = "postgres",
        database_url = "postgresql://test:test@localhost:5432/test",
    )

    with open_checkpointer(settings) as checkpointer:
        assert checkpointer is fake_checkpointer
        assert fake_context.enter_called is True
        assert fake_checkpointer.setup_called is True
        assert fake_context.exit_called is False

    assert fake_context.exit_called is True
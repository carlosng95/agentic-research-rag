from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from .config import Settings


@contextmanager
def open_checkpointer(settings: Settings) -> Iterator[BaseCheckpointSaver]:
    if settings.checkpoint_backend == "memory":
        yield InMemorySaver()
        return

    if settings.checkpoint_backend == "postgres":
        with PostgresSaver.from_conn_string(settings.database_url) as checkpointer:
            checkpointer.setup()
            yield checkpointer
        return

    raise ValueError(
        f"Unsupported checkpoint backend: {settings.checkpoint_backend}"
    )
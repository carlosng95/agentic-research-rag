from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from .context import get_request_id
from .metrics import emit_metric


def _metric_properties() -> dict[str, object]:
    request_id = get_request_id()

    if request_id is None:
        return {}

    return {
        "request_id": request_id,
    }


@contextmanager
def observe_operation(name: str) -> Iterator[None]:
    started_at = perf_counter()

    emit_metric(
        "OperationCount",
        1,
        dimensions = {
            "Operation": name,
        },
        properties = _metric_properties(),
    )

    try:
        yield

    except Exception:
        emit_metric(
            "OperationErrorCount",
            1,
            dimensions = {
                "Operation": name,
            },
            properties = _metric_properties(),
        )

        raise

    finally:
        duration_ms = (perf_counter() - started_at) * 1000

        emit_metric(
            "OperationLatency",
            duration_ms,
            unit = "Milliseconds",
            dimensions = {
                "Operation": name,
            },
            properties = _metric_properties(),
        )
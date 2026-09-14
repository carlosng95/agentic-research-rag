from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import Any, TypeVar, cast

from .context import get_request_id
from .metrics import emit_metric


NodeCallable = TypeVar(
    "NodeCallable",
    bound = Callable[..., Any],
)


def _metric_properties() -> dict[str, object]:
    request_id = get_request_id()

    if request_id is None:
        return {}

    return {
        "request_id": request_id,
    }


def observe_node(
    name: str,
    node: NodeCallable,
) -> NodeCallable:
    if inspect.iscoroutinefunction(node):
        @wraps(node)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            started_at = perf_counter()

            emit_metric(
                "GraphNodeCount",
                1,
                dimensions = {
                    "Node": name,
                },
                properties = _metric_properties(),
            )

            try:
                return await node(
                    *args,
                    **kwargs,
                )

            except Exception:
                emit_metric(
                    "GraphNodeErrorCount",
                    1,
                    dimensions = {
                        "Node": name,
                    },
                    properties = _metric_properties(),
                )

                raise

            finally:
                duration_ms = (perf_counter() - started_at) * 1000

                emit_metric(
                    "GraphNodeLatency",
                    duration_ms,
                    unit = "Milliseconds",
                    dimensions = {
                        "Node": name,
                    },
                    properties = _metric_properties(),
                )

        return cast(
            NodeCallable,
            async_wrapper,
        )

    @wraps(node)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started_at = perf_counter()

        emit_metric(
            "GraphNodeCount",
            1,
            dimensions = {
                "Node": name,
            },
            properties = _metric_properties(),
        )

        try:
            return node(
                *args,
                **kwargs,
            )

        except Exception:
            emit_metric(
                "GraphNodeErrorCount",
                1,
                dimensions = {
                    "Node": name,
                },
                properties = _metric_properties(),
            )

            raise

        finally:
            duration_ms = (perf_counter() - started_at) * 1000

            emit_metric(
                "GraphNodeLatency",
                duration_ms,
                unit = "Milliseconds",
                dimensions = {
                    "Node": name,
                },
                properties = _metric_properties(),
            )

    return cast(
        NodeCallable,
        wrapper,
    )
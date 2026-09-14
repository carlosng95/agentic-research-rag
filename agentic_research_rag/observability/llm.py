from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable

from .context import get_request_id
from .metrics import emit_metric


_OPERATION_TAG_PREFIX = "llm_operation:"


@dataclass(frozen = True)
class _LLMRun:
    operation: str
    started_at: float
    request_id: str | None


def with_llm_operation(runnable: Runnable, operation: str) -> Runnable:
    return runnable.with_config(
        {
            "run_name": f"llm_{operation}",
            "tags": [
                f"{_OPERATION_TAG_PREFIX}{operation}",
            ],
        }
    )


class LLMMetricsCallback(BaseCallbackHandler):
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._runs: dict[UUID, _LLMRun] = {}
        self._lock = Lock()

    def _get_operation(self, tags: list[str] | None) -> str:
        for tag in tags or []:
            if tag.startswith(_OPERATION_TAG_PREFIX):
                return tag.removeprefix(_OPERATION_TAG_PREFIX)

        return "unknown"

    def _start_run(
        self,
        run_id: UUID,
        tags: list[str] | None,
    ) -> None:
        run = _LLMRun(
            operation = self._get_operation(tags),
            started_at = perf_counter(),
            request_id = get_request_id(),
        )

        with self._lock:
            self._runs[run_id] = run

        emit_metric(
            "LLMCallCount",
            1,
            dimensions = {
                "Operation": run.operation,
                "Model": self._model_name,
            },
            properties = self._properties(
                run = run,
                run_id = run_id,
            ),
        )

    def _pop_run(self, run_id: UUID) -> _LLMRun | None:
        with self._lock:
            return self._runs.pop(
                run_id,
                None,
            )

    def _properties(
        self,
        run: _LLMRun,
        run_id: UUID,
    ) -> dict[str, object]:
        properties: dict[str, object] = {
            "llm_run_id": str(run_id),
        }

        if run.request_id is not None:
            properties["request_id"] = run.request_id

        return properties

    def _emit_latency(
        self,
        run: _LLMRun,
        run_id: UUID,
    ) -> None:
        duration_ms = (
            perf_counter() - run.started_at
        ) * 1000

        emit_metric(
            "LLMCallLatency",
            duration_ms,
            unit = "Milliseconds",
            dimensions = {
                "Operation": run.operation,
                "Model": self._model_name,
            },
            properties = self._properties(
                run = run,
                run_id = run_id,
            ),
        )

    def _extract_usage(
        self,
        response: LLMResult,
    ) -> tuple[int, int, int] | None:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        usage_found = False

        for generation_group in response.generations:
            for generation in generation_group:
                message = getattr(
                    generation,
                    "message",
                    None,
                )

                usage = getattr(
                    message,
                    "usage_metadata",
                    None,
                )

                if not usage:
                    continue

                usage_found = True
                input_tokens += int(
                    usage.get(
                        "input_tokens",
                        0,
                    )
                    or 0
                )

                output_tokens += int(
                    usage.get(
                        "output_tokens",
                        0,
                    )
                    or 0
                )

                total_tokens += int(
                    usage.get(
                        "total_tokens",
                        0,
                    )
                    or 0
                )

        if usage_found:
            return (
                input_tokens,
                output_tokens,
                total_tokens,
            )

        llm_output = response.llm_output or {}
        token_usage = llm_output.get(
            "token_usage",
            {},
        )

        if not token_usage:
            return None

        input_tokens = int(
            token_usage.get(
                "prompt_tokens",
                0,
            )
            or 0
        )

        output_tokens = int(
            token_usage.get(
                "completion_tokens",
                0,
            )
            or 0
        )

        total_tokens = int(
            token_usage.get(
                "total_tokens",
                input_tokens + output_tokens,
            )
            or 0
        )

        return (
            input_tokens,
            output_tokens,
            total_tokens,
        )

    def _emit_usage(
        self,
        run: _LLMRun,
        run_id: UUID,
        response: LLMResult,
    ) -> None:
        usage = self._extract_usage(
            response = response
        )

        if usage is None:
            return

        input_tokens, output_tokens, total_tokens = usage

        dimensions = {
            "Operation": run.operation,
            "Model": self._model_name,
        }

        properties = self._properties(
            run = run,
            run_id = run_id,
        )

        emit_metric(
            "LLMInputTokens",
            input_tokens,
            dimensions = dimensions,
            properties = properties,
        )

        emit_metric(
            "LLMOutputTokens",
            output_tokens,
            dimensions = dimensions,
            properties = properties,
        )

        emit_metric(
            "LLMTotalTokens",
            total_tokens,
            dimensions = dimensions,
            properties = properties,
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_run(
            run_id = run_id,
            tags = tags,
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_run(
            run_id = run_id,
            tags = tags,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._pop_run(
            run_id = run_id
        )

        if run is None:
            return

        self._emit_latency(
            run = run,
            run_id = run_id,
        )

        self._emit_usage(
            run = run,
            run_id = run_id,
            response = response,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._pop_run(
            run_id = run_id
        )

        if run is None:
            return

        self._emit_latency(
            run = run,
            run_id = run_id,
        )

        emit_metric(
            "LLMCallErrorCount",
            1,
            dimensions = {
                "Operation": run.operation,
                "Model": self._model_name,
            },
            properties = self._properties(
                run = run,
                run_id = run_id,
            ),
        )
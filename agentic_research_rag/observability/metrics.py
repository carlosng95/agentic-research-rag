from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping
from typing import Literal


MetricUnit = Literal[
    "Count",
    "Milliseconds",
    "Seconds",
    "Bytes",
    "Percent",
    "None",
]


def emit_metric(
    name: str,
    value: int | float,
    unit: MetricUnit = "Count",
    *,
    dimensions: Mapping[str, str] | None = None,
    properties: Mapping[str, object] | None = None,
) -> None:

    namespace = os.getenv("METRICS_NAMESPACE", "AgenticResearchRAG")
    service = os.getenv("SERVICE_NAME", "agentic-research-rag")
    environment = os.getenv("ENVIRONMENT", "local")

    metric_dimensions = {
        "Service": service,
        "Environment": environment,
    }

    if dimensions:
        metric_dimensions.update(dimensions)

    payload: dict[str, object] = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": namespace,
                    "Dimensions": [list(metric_dimensions)],
                    "Metrics": [
                        {
                            "Name": name,
                            "Unit": unit,
                        }
                    ],
                }
            ],
        },
        **metric_dimensions,
        name: value,
    }

    if properties:
        payload.update(properties)

    sys.stdout.write(json.dumps(payload, separators = (",", ":")) + "\n")
    sys.stdout.flush()
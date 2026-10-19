import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any, TextIO


class ConsoleLogger:
    """Logs metrics to the shell in a readable per-step format."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def log(self, metrics: dict[str, Any], step: int) -> None:
        epoch = metrics.get("epoch", step)
        print(f"\nIteration {epoch}", file=self._stream)

        for key, value in self._flatten_metrics(metrics):
            print(f"  {key}: {self._format_value(value)}", file=self._stream)

        self._stream.flush()

    def finish(self) -> None:
        self._stream.flush()

    def _flatten_metrics(
        self,
        metrics: Mapping[str, Any],
        prefix: str = "",
    ) -> list[tuple[str, Any]]:
        items: list[tuple[str, Any]] = []

        for key, value in metrics.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, Mapping):
                items.extend(self._flatten_metrics(value, full_key))
            else:
                items.append((full_key, value))

        return items

    def _format_value(self, value: Any) -> str:
        if isinstance(value, (int, bool)):
            return str(value)

        if isinstance(value, float):
            return f"{value:.6f}"

        if value is None:
            return "None"

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return json.dumps(value)

        return str(value)

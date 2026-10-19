import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonLogger:
    """Logs metrics as JSON lines to a file."""

    def __init__(self, log_dir: Path) -> None:
        self._path = log_dir / "metrics.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        record = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def finish(self) -> None:
        pass
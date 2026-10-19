from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class RunResult:
    checkpoint_path: Path
    config: dict[str, Any]
    metrics: dict[str, Any]
    from_cache: bool = False


@runtime_checkable
class Runner(Protocol):
    def run(
        self,
        trainer_class: type,
        config: dict[str, Any],
        epochs: int,
        checkpoint_frequency: int,
        storage_dir: Path,
    ) -> RunResult: ...
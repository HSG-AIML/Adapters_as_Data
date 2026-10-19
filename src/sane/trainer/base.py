from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Trainer(Protocol):
    """Interface that all trainers must implement."""

    epoch: int

    def setup(self, config: dict[str, Any]) -> None:
        """Initialize the trainer from a config."""
        ...

    def step(self) -> dict[str, Any]:
        """Run a single training step and return metrics."""
        ...

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """Persist current state to disk."""
        ...

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Restore state from a checkpoint on disk."""
        ...
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """Logs scalar metrics to TensorBoard."""

    def __init__(self, log_dir: Path) -> None:
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(log_dir))
        except ImportError:
            logger.warning(
                "TensorBoard is not available. Install tensorboard to enable logging."
            )

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self.writer is None:
            return
        for tag, value in self._flatten_metrics(metrics):
            self.writer.add_scalar(tag, value, step)

    def _flatten_metrics(
        self,
        metrics: Mapping[str, Any],
        prefix: str = "",
    ) -> list[tuple[str, float]]:
        items: list[tuple[str, float]] = []

        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key

            if isinstance(value, Mapping):
                items.extend(self._flatten_metrics(value, tag))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                items.append((tag, float(value)))

        return items

    def finish(self) -> None:
        if self.writer is not None:
            self.writer.close()
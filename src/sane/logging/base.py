from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetricsLogger(Protocol):
    def log(self, metrics: dict[str, Any], step: int) -> None: ...
    def finish(self) -> None: ...


class CompositeLogger:
    def __init__(self, loggers: list[MetricsLogger]) -> None:
        self._loggers = loggers

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for logger in self._loggers:
            logger.log(metrics, step)

    def finish(self) -> None:
        for logger in self._loggers:
            logger.finish()
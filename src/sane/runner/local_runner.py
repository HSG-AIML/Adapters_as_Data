import itertools
import json
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from sane.logging.base import MetricsLogger
from sane.runner.base import RunResult


class LocalRunner:
    def __init__(self, logger: MetricsLogger | None = None) -> None:
        self._logger = logger

    def run(
        self,
        trainer_class: type,
        config: dict[str, Any],
        epochs: int,
        checkpoint_frequency: int,
        storage_dir: Path,
    ) -> RunResult:
        configs = expand_grid(config)
        results: list[RunResult] = []
        for cfg in configs:
            result = self._run_single(
                trainer_class,
                cfg,
                epochs,
                checkpoint_frequency,
                storage_dir,
            )
            results.append(result)
        return min(results, key=lambda r: r.metrics.get("loss", float("inf")))

    def _run_single(
        self,
        trainer_class: type,
        config: dict[str, Any],
        epochs: int,
        checkpoint_frequency: int,
        storage_dir: Path,
    ) -> RunResult:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        trial_dir = storage_dir / f"{trainer_class.__name__}_{timestamp}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        logger = self._logger or _build_default_logger(trial_dir)

        try:
            from omegaconf import OmegaConf

            # Add trial directory to config (used in trainer setup)
            config["trial_dir"] = str(trial_dir)
            OmegaConf.save(OmegaConf.create(config), str(trial_dir / "trial.yaml"))
            _save_trial_configs(config, trial_dir)

            trainer = trainer_class()
            trainer.setup(config)

            metrics: dict[str, Any] = {}
            for epoch in range(1, epochs + 1):
                metrics = trainer.step()
                logger.log(metrics, step=epoch)

                is_last = epoch == epochs
                if epoch % checkpoint_frequency == 0 or is_last:
                    ckpt_dir = trial_dir / f"checkpoint_{epoch:06d}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    trainer.save_checkpoint(str(ckpt_dir))

            final_ckpt_path = trial_dir / f"checkpoint_{epochs:06d}"
            return RunResult(
                checkpoint_path=final_ckpt_path,
                config=config,
                metrics=metrics,
            )

        finally:
            logger.finish()


def expand_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries = _collect_grid_entries(config)
    if not entries:
        return [config]

    paths, value_lists = zip(*entries)
    configs: list[dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        cfg = deepcopy(config)
        for path, value in zip(paths, combo):
            _set_nested(cfg, path, value)
        configs.append(cfg)
    return configs



def _save_trial_configs(config: dict[str, Any], trial_dir: Path) -> None:
    from omegaconf import OmegaConf

    OmegaConf.save(OmegaConf.create(config), str(trial_dir / "config.yaml"))
    (trial_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str)
    )


def _build_default_logger(log_dir: Path) -> MetricsLogger:
    from sane.logging import CompositeLogger, JsonLogger, TensorBoardLogger, ConsoleLogger

    return CompositeLogger([JsonLogger(log_dir), TensorBoardLogger(log_dir), ConsoleLogger()])


def _to_none(x: Any) -> Any:
    return None if x in ("none", "None") else x


def _collect_grid_entries(
    node: Any, prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], list[Any]]]:
    """Walk config recursively, collecting (path, values) for each grid spec."""
    entries: list[tuple[tuple[str, ...], list[Any]]] = []
    if isinstance(node, Mapping) and len(node) == 1 and next(iter(node)) == "grid":
        grid_values = cast(list[Any], list(node.values())[0])
        values = [_to_none(v) for v in grid_values]
        entries.append((prefix, values))
    elif isinstance(node, MutableMapping):
        for k, v in node.items():
            entries.extend(_collect_grid_entries(v, prefix + (k,)))
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for i, v in enumerate(node):
            entries.extend(_collect_grid_entries(v, prefix + (str(i),)))
    return entries


def _set_nested(d: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set a value in a nested dict/list structure given a key path."""
    obj: Any = d
    for key in path[:-1]:
        if isinstance(obj, list):
            obj = obj[int(key)]
        else:
            obj = obj[key]
    last = path[-1]
    if isinstance(obj, list):
        obj[int(last)] = value
    else:
        obj[last] = value
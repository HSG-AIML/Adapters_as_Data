from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from sane.runner.base import RunResult

logger = logging.getLogger(__name__)


class Orchestrator:
    """Sequences training stages and delegates execution to a Runner."""

    def __init__(
        self,
        config: dict,
        force_local: bool = False,
    ) -> None:
        self._config = config
        root_dir = Path(config["root_dir"])
        experiment_dir = root_dir / config["experiment_name"]
        experiment_dir.mkdir(parents=True, exist_ok=True)
        storage_dir = experiment_dir
        self._storage_dir = storage_dir
        logger.debug("Storage directory: %s", storage_dir)
        self._runner = _build_runner(config, force_local=force_local)
        logger.debug("Runner: %s (force_local=%s)", type(self._runner).__name__, force_local)

    def run(self) -> RunResult:
        cfg = self._config
        self._save_config()
        logger.debug("Config saved to %s", self._storage_dir / "config.yaml")

        trainer_class = _resolve_trainer_class(cfg)
        epochs = cfg["epochs"]
        checkpoint_frequency = cfg["checkpoint_frequency"]

        logger.info(
            "Starting experiment %s (%s, %d epochs)",
            cfg["experiment_name"],
            trainer_class.__name__,
            epochs,
        )

        logger.debug("Delegating to %s", type(self._runner).__name__)
        result = self._runner.run(
            trainer_class=trainer_class,
            config=cfg,
            epochs=epochs,
            checkpoint_frequency=checkpoint_frequency,
            storage_dir=self._storage_dir,
        )

        logger.info("Experiment finished. Checkpoint: %s", result.checkpoint_path)
        return result

    def _save_config(self) -> None:
        config_path = self._storage_dir / f"config_{_config_hash(self._config)}.yaml"
        OmegaConf.save(OmegaConf.create(self._config), str(config_path))


def _build_runner(
    config: dict[str, Any],
    force_local: bool = False,
):
    runner_cfg = config["runner"]
    target = _resolve_target(runner_cfg, _RUNNER_ALIASES)

    is_ray = target == _RUNNER_ALIASES["ray"]
    if is_ray and not force_local:
        runner_cls = _get_class(target)
        if isinstance(runner_cfg, dict):
            # Future _target_-style: all kwargs live inside the runner sub-dict.
            return runner_cls(
                num_cpus=runner_cfg["num_cpus"],
                num_gpus=runner_cfg["num_gpus"],
                cpus_per_trial=runner_cfg["cpus_per_trial"],
                gpus_per_trial=runner_cfg["gpus_per_trial"],
                num_samples=runner_cfg["num_samples"],
                num_to_keep=runner_cfg["num_to_keep"],
                verbose=runner_cfg["verbose"],
                wandb_project=runner_cfg["wandb_project"],
                wandb_group=runner_cfg["wandb_group"],
                wandb_api_key_file=runner_cfg["wandb_api_key_file"],
            )
        else:
            # Legacy flat config: Ray kwargs are top-level keys.
            return runner_cls(
                num_cpus=config["num_cpus"],
                num_gpus=config["num_gpus"],
                cpus_per_trial=config["cpu_per_trial"],
                gpus_per_trial=config["gpu_per_trial"],
                num_samples=config["num_samples"],
                num_to_keep=config["num_checkpoints_to_keep"],
                verbose=config["ray_verbose"],
                wandb_project=config["wandb_project"],
                wandb_group=config["wandb_group"],
                wandb_api_key_file=config["wandb_api_key_file"],
            )

    local_target = _RUNNER_ALIASES["local"] if force_local else target
    runner_cls = _get_class(local_target)
    return runner_cls()


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _resolve_trainer_class(cfg: dict[str, Any] | str) -> type:
    """Resolve the trainer class from the config.

    Supports a plain name (``"SANETrainer"``), a full config dict
    containing a ``trainer`` key, or a Hydra-style sub-dict with
    ``_target_``.
    """
    if isinstance(cfg, str):
        trainer_cfg: dict[str, Any] | str = cfg
    else:
        trainer_cfg = cfg["trainer"]
    target = _resolve_target(trainer_cfg, _TRAINER_ALIASES)
    if not target:
        raise ValueError(f"Cannot resolve trainer from config: {trainer_cfg!r}")
    try:
        return _get_class(target)
    except Exception as exc:
        raise ValueError(f"Unknown trainer class: {target!r}") from exc


def _resolve_target(cfg: dict[str, Any] | str, aliases: dict[str, str]) -> str:
    """Extract a fully-qualified ``_target_`` from *cfg*.

    If *cfg* is a dict containing ``_target_``, return it directly.
    Otherwise treat *cfg* (or the string itself) as a short alias and
    resolve it through *aliases*, falling back to treating the value as
    a dotpath.
    """
    if isinstance(cfg, dict) and "_target_" in cfg:
        return cfg["_target_"]
    name = cfg if isinstance(cfg, str) else ""
    if name in aliases:
        return aliases[name]
    # Assume the caller passed a fully-qualified dotpath already.
    return name


def _get_class(target: str) -> type:
    """Resolve a dotpath to a class via Hydra's lookup machinery."""
    from hydra.utils import get_class

    return get_class(target)


# ---------------------------------------------------------------------------
# Alias constants – legacy short names to fully-qualified dotpaths.
# These can be removed once all configs use ``_target_`` directly.
# ---------------------------------------------------------------------------

_TRAINER_ALIASES: dict[str, str] = {
    "SANETrainer": "sane.trainer.SANETrainer",
}

_RUNNER_ALIASES: dict[str, str] = {
    "local": "sane.runner.local_runner.LocalRunner",
    "ray": "sane.runner.ray_runner.RayRunner",
}


def _config_hash(config: dict[str, Any]) -> str:
    """Return an 8-character hex digest of the config, excluding root_dir."""
    hashable = {k: v for k, v in config.items() if k != "root_dir"}
    serialized = json.dumps(hashable, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:8]

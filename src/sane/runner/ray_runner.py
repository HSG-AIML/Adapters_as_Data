import json
import logging
import tempfile
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

import ray
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import Checkpoint

from sane.runner.base import RunResult
from sane.utils.generic_utils import log_mem

logger = logging.getLogger(__name__)


class RayRunner:
    """Runner that delegates training to Ray Tune."""

    def __init__(
        self,
        num_cpus: int,
        num_gpus: int,
        cpus_per_trial: int,
        gpus_per_trial: float,
        num_samples: int,
        num_to_keep: int,
        verbose: int,
        wandb_project: str | None = None,
        wandb_group: str | None = None,
        wandb_api_key_file: str | None = None,
    ) -> None:
        self._num_cpus = num_cpus
        self._num_gpus = num_gpus
        self._cpus_per_trial = cpus_per_trial
        self._gpus_per_trial = gpus_per_trial
        self._num_samples = num_samples
        self._num_to_keep = num_to_keep
        self._verbose = verbose
        self._wandb_project = wandb_project
        self._wandb_group = wandb_group
        self._wandb_api_key_file = wandb_api_key_file

    def run(
        self,
        trainer_class: type,
        config: dict[str, Any],
        epochs: int,
        checkpoint_frequency: int,
        storage_dir: Path,
    ) -> RunResult:
        we_initialized = not ray.is_initialized()
        ray.init(
            num_cpus=self._num_cpus,
            num_gpus=self._num_gpus,
            ignore_reinit_error=True,
        )
        if we_initialized:
            logger.info("Ray initialized: %s", ray.cluster_resources())

        try:
            return self._run_tuner(
                trainer_class, config, epochs, checkpoint_frequency, storage_dir, resume_path=config.get("resume_path"),
            )
        finally:
            if we_initialized:
                ray.shutdown()

    def _run_tuner(
        self,
        trainer_class: type,
        config: dict[str, Any],
        epochs: int,
        checkpoint_frequency: int,
        storage_dir: Path,
        resume_path: str | None = None,
    ) -> RunResult:
        param_space = _inject_ray_grids(config)

        train_func = _make_trainable(trainer_class, epochs, checkpoint_frequency)
        trainable = tune.with_resources(
            train_func,
            resources={"cpu": self._cpus_per_trial, "gpu": self._gpus_per_trial},
        )

        experiment_name = config["experiment_name"]

        callbacks = []
        if self._wandb_project:
            callbacks.append(WandbLoggerCallback(
                project=self._wandb_project,
                group=self._wandb_group,
                api_key_file=self._wandb_api_key_file,
            ))

        resolved = storage_dir.resolve()

        if resume_path:
            logger.info("Restoring experiment from %s", resume_path)
            tuner = tune.Tuner.restore(
                path=resume_path,
                trainable=trainable,
                resume_unfinished=True,
                resume_errored=False,
            )
        else:
            tuner = tune.Tuner(
                trainable,
                param_space=param_space,
                tune_config=tune.TuneConfig(
                    num_samples=self._num_samples,
                    reuse_actors=False,
                ),
                run_config=tune.RunConfig(
                    name=experiment_name,
                    storage_path=str(resolved.parent),
                    checkpoint_config=tune.CheckpointConfig(
                        num_to_keep=self._num_to_keep,
                    ),
                    verbose=self._verbose,
                    callbacks=callbacks or None,
                ),
            )

        result_grid = tuner.fit()
        best = result_grid.get_best_result(metric="loss", mode="min")

        best_checkpoint = best.checkpoint
        if best_checkpoint is not None:
            checkpoint_path = Path(best_checkpoint.path)
        else:
            checkpoint_path = storage_dir

        best_config = best.config or config
        best_metrics = dict(best.metrics) if best.metrics else {}

        return RunResult(
            checkpoint_path=checkpoint_path,
            config=best_config,
            metrics=best_metrics,
        )


def _make_trainable(trainer_class: type, epochs: int, checkpoint_frequency: int):
    """Build a Ray Tune function trainable that wraps a plain trainer."""

    def train_func(config: dict) -> None:

        import os
        import torch

        print("PYTHON:", os.environ.get("VIRTUAL_ENV"))
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("TORCH:", torch.__version__)
        print("TORCH CUDA:", torch.version.cuda)
        print("CUDA AVAILABLE:", torch.cuda.is_available())

        log_mem("Start of train_func in runner")

        # Get trial directory and add to config (used in trainer setup)
        trial_dir = Path(tune.get_context().get_storage().trial_fs_path)
        config["trial_dir"] = str(trial_dir)
        # Save the resolved per-trial config (grid values already expanded by Ray)
        OmegaConf.save(OmegaConf.create(config), str(trial_dir / "trial.yaml"))
        OmegaConf.save(OmegaConf.create(config), str(trial_dir / "config.yaml"))
        (trial_dir / "config.json").write_text(
            json.dumps(config, indent=2, default=str)
        )

        start_epoch = 1
        trainer = trainer_class()
        trainer.setup(config)

        checkpoint = tune.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as ckpt_dir:
                trainer.load_checkpoint(ckpt_dir)
                start_epoch = trainer.epoch + 1

        for epoch in range(start_epoch, epochs + 1):
            metrics = trainer.step()

            should_checkpoint = (
                checkpoint_frequency > 0 and epoch % checkpoint_frequency == 0
            ) or epoch == epochs

            if should_checkpoint:
                with tempfile.TemporaryDirectory() as tmpdir:
                    trainer.save_checkpoint(tmpdir)
                    tune.report(
                        metrics=metrics,
                        checkpoint=Checkpoint.from_directory(tmpdir),
                    )
            else:
                tune.report(metrics=metrics)

    return train_func


def _inject_ray_grids(node: Any) -> Any:
    """Convert ``{"grid": [v1, v2, ...]}`` specs to ``tune.grid_search``."""
    if isinstance(node, Mapping) and len(node) == 1 and next(iter(node)) == "grid":
        values = list(node.values())[0]
        values = [None if v in ("none", "None") else v for v in values]
        return tune.grid_search(values)
    if isinstance(node, MutableMapping):
        return {k: _inject_ray_grids(v) for k, v in node.items()}
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        t = type(node)
        try:
            return t(_inject_ray_grids(v) for v in node)
        except TypeError:
            return [_inject_ray_grids(v) for v in node]
    return node

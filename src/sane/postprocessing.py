"""Shared infrastructure for SANE post-training artifact scripts."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from sane.data.config_schema import (
    CachedWindowedDatasetConfig,
    CombinedLoRAWeightsDatasetConfig,
    DataLoaderConfig,
    DatasetFactoryConfig,
    LoRAWeightsDatasetConfig,
    SplitConfig,
    TokenizerConfig,
    from_dict,
)
from sane.data.dataset_factory import DatasetFactory
from sane.model.autoencoder import SANEAutoEncoder


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A reconstructed SANE model and its checkpoint metadata."""

    model: SANEAutoEncoder
    epoch: int
    config: dict[str, Any]
    path: Path


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` and validate explicitly requested CUDA devices."""

    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested, but CUDA is unavailable")
    return device


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML or JSON configuration as a plain mapping."""

    path = Path(path)
    with path.open(encoding="utf-8") as config_file:
        if path.suffix.lower() in {".yaml", ".yml"}:
            config = yaml.safe_load(config_file)
        elif path.suffix.lower() == ".json":
            config = json.load(config_file)
        else:
            raise ValueError(f"Config must be YAML or JSON: {path}")
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def load_autoencoder_checkpoint(
    path: Path | str,
    device: torch.device,
) -> LoadedCheckpoint:
    """Load a trusted SANE checkpoint and reconstruct its autoencoder."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping: {path}")
    try:
        epoch = int(state["epoch"])
        config = copy.deepcopy(state["config"])
        model_config = copy.deepcopy(config["model"])
        model_state = state["model"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Checkpoint must contain epoch, config.model, and model fields"
        ) from error
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a dictionary")

    model = SANEAutoEncoder.from_config({**model_config, "device": str(device)})
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return LoadedCheckpoint(model=model, epoch=epoch, config=config, path=path)


def choose_data_config(
    *,
    explicit_path: Path | None,
    checkpoint_config: dict[str, Any],
    payload_config: Any = None,
) -> tuple[dict[str, Any], str]:
    """Resolve data config as explicit override, payload config, then checkpoint."""

    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Config not found: {explicit_path}")
        return load_config(explicit_path), str(explicit_path.resolve())
    if isinstance(payload_config, dict) and "data" in payload_config:
        return payload_config, "payload:config"
    if isinstance(checkpoint_config, dict) and "data" in checkpoint_config:
        return checkpoint_config, "checkpoint:config"
    raise ValueError(
        "No complete data config was found in the payload or checkpoint. "
        "Pass --config explicitly."
    )


def _resolve_checkpoint_dataset(config: Any) -> Any:
    if isinstance(config, list):
        return [_resolve_checkpoint_dataset(item) for item in config]

    if isinstance(config, str):
        import sane
        from omegaconf import OmegaConf

        config_path = (
            Path(sane.__file__).resolve().parents[2]
            / "config"
            / "data"
            / "checkpoints_dataset"
            / f"{config}.yaml"
        )
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint dataset config alias {config!r} was not found at {config_path}"
            )
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid checkpoints_dataset config: {config!r}")
    class_name = config.get("class")
    if class_name == "LoRAWeightsDataset":
        return from_dict(LoRAWeightsDatasetConfig, config)
    if class_name == "CombinedLoRAWeightsDataset":
        datasets = [
            from_dict(LoRAWeightsDatasetConfig, item)
            for item in config.get("datasets", [])
        ]
        if not datasets:
            raise ValueError("CombinedLoRAWeightsDataset requires at least one dataset")
        return CombinedLoRAWeightsDatasetConfig(datasets=datasets)
    raise ValueError(f"Unsupported checkpoints dataset class: {class_name!r}")


def build_factory_config(config: dict[str, Any]) -> DatasetFactoryConfig:
    """Convert a resolved experiment config into DatasetFactory's schema."""

    try:
        data_config = config["data"]
        token_dataset_config = copy.deepcopy(data_config["tokens_dataset"])
        token_dataset_config["data_config_for_hash"] = copy.deepcopy(data_config)
        return DatasetFactoryConfig(
            checkpoints_dataset=_resolve_checkpoint_dataset(
                data_config["checkpoints_dataset"]
            ),
            tokenizer=from_dict(TokenizerConfig, data_config["tokenizer"]),
            tokens_dataset=from_dict(
                CachedWindowedDatasetConfig,
                token_dataset_config,
            ),
            split=from_dict(SplitConfig, data_config["split"]),
            dataloader=from_dict(DataLoaderConfig, data_config),
        )
    except KeyError as error:
        raise ValueError(f"Incomplete data configuration; missing {error}") from error


def build_dataloaders(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Build train, validation, and test loaders from a resolved config."""

    return DatasetFactory().build(build_factory_config(config))


def tokenizer_from_dataloaders(dataloaders: tuple[Any, Any, Any]) -> Any:
    """Return the shared tokenizer attached to the train dataset."""

    train_loader = dataloaders[0]
    if train_loader is None:
        raise ValueError("A train split is required to construct the tokenizer")
    return train_loader.dataset.tokenizer


@torch.no_grad()
def decode_latents(
    model: SANEAutoEncoder,
    latents: torch.Tensor,
    positions: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a latent batch and return token grids plus matching positions."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    decoded_chunks = []
    position_chunks = []
    for start in tqdm(range(0, latents.shape[0], batch_size), desc="Decoding"):
        z_batch = latents[start : start + batch_size].to(device)
        p_batch = positions[start : start + batch_size].to(device)
        if model.translator is not None:
            z_batch = model.translator(z_batch)
            if z_batch.shape[1] != p_batch.shape[1]:
                token_indices = torch.arange(z_batch.shape[1], device=device)
                p_batch = torch.stack(
                    (token_indices, torch.zeros_like(token_indices), token_indices),
                    dim=1,
                ).unsqueeze(0).repeat(z_batch.shape[0], 1, 1)
                p_batch = p_batch.to(dtype=positions.dtype)
        decoded_chunks.append(model.decoder(z_batch, p_batch).cpu())
        position_chunks.append(p_batch.cpu())
    return torch.cat(decoded_chunks).float(), torch.cat(position_chunks)


def flatten_token_grid(
    token_grid: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    tokenizer: Any,
    *,
    description: str,
) -> torch.Tensor:
    """Undo token slicing and flatten each reconstructed state dictionary."""

    if token_grid.shape[:2] != mask.shape or token_grid.shape[:2] != positions.shape[:2]:
        raise ValueError(
            f"Token grid, mask, and positions are not aligned: "
            f"{tuple(token_grid.shape)}, {tuple(mask.shape)}, {tuple(positions.shape)}"
        )
    flat_weights = []
    for row in tqdm(range(token_grid.shape[0]), desc=description, leave=False):
        segments = tokenizer.unslice(
            token_grid[row],
            mask=mask[row],
            position=positions[row],
        )
        state_dict = tokenizer.rebuild_state_dict(segments)
        flat_weights.append(
            torch.cat([tensor.reshape(-1) for tensor in state_dict.values()])
        )
    return torch.stack(flat_weights)

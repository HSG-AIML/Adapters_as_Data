"""Decode SANE latent payloads into flattened LoRA weight payloads.

This is the second half of the post-training artifact bridge:

    latents .pt -> SANE decoder -> de-tokenization -> flat x / x_hat .pt

The output can be passed directly to ``experiments/generate_images.py`` or to any
weight-space control script. The ``x`` field is the source/reference adapter
stored in the latent payload; for generated latents it may intentionally be only
a compatibility reference, so render those outputs with ``--fields x_hat
--no-scores``.

Examples:

    uv run python experiments/reconstruct_latents.py \
        path/to/checkpoint.pt outputs/latents/latents_val.pt \
        --output-dir outputs/reconstructions

    uv run python experiments/reconstruct_latents.py \
        path/to/checkpoint.pt outputs/editing/latents/*.pt \
        --output-dir outputs/editing/reconstructed

The checkpoint and latent payloads are pickle-backed and must come from trusted
sources. The checkpoint's decoder is always used; ``--config`` only overrides
data/tokenizer configuration.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from sane.postprocessing import (
    build_dataloaders,
    choose_data_config,
    decode_latents,
    flatten_token_grid,
    load_autoencoder_checkpoint,
    resolve_device,
    tokenizer_from_dataloaders,
)


FORMAT_NAME = "sane.reconstructed_weights"
FORMAT_VERSION = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Trusted SANE checkpoint.pt file.")
    parser.add_argument(
        "latents",
        nargs="+",
        type=Path,
        help="One or more trusted latent payloads to reconstruct.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-name",
        help="Output filename override; valid only with one latent input.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML/JSON data/tokenizer config override.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. cuda:0 or cpu (default: auto).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing reconstruction payloads.",
    )
    return parser.parse_args(argv)


def default_output_name(latents_path: Path) -> str:
    """Choose conventional names while preserving experiment-specific stems."""

    if latents_path.stem.startswith("latents_"):
        suffix = latents_path.stem.removeprefix("latents_")
        return f"x_and_x_hat_{suffix}.pt"
    return latents_path.name


def output_paths(args: argparse.Namespace) -> list[Path]:
    if args.output_name is not None:
        if len(args.latents) != 1:
            raise ValueError("--output-name can only be used with one latent input")
        name = args.output_name
        if Path(name).name != name:
            raise ValueError("--output-name must be a filename, not a path")
        names = [name]
    else:
        names = [default_output_name(path) for path in args.latents]

    paths = [args.output_dir / name for name in names]
    if len(set(paths)) != len(paths):
        raise ValueError(
            "Multiple latent inputs resolve to the same output filename; run them "
            "separately with --output-name or use distinct input names"
        )
    return paths


def load_latent_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Latent payload not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Latent payload must contain a mapping: {path}")
    return dict(payload)


def validate_latent_payload(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    missing = [field for field in ("z", "x", "p") if field not in payload]
    if missing:
        raise KeyError(f"{path} is missing required fields: {missing}")

    latents = payload["z"]
    source_tokens = payload["x"]
    positions = payload["p"]
    mask = payload.get("mask")
    for field, tensor in (
        ("z", latents),
        ("x", source_tokens),
        ("p", positions),
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"{path}: {field} must be a rank-3 tensor")

    if latents.shape[:2] != source_tokens.shape[:2]:
        raise ValueError(f"{path}: z and x must align on samples and tokens")
    if positions.shape[:2] != latents.shape[:2]:
        raise ValueError(f"{path}: p must align with z on samples and tokens")
    if positions.shape[2] != 3:
        raise ValueError(f"{path}: p must have shape [N, tokens, 3]")

    if mask is None:
        mask = torch.ones(latents.shape[:2], dtype=torch.bool)
    elif not isinstance(mask, torch.Tensor) or mask.shape != latents.shape[:2]:
        raise ValueError(f"{path}: mask must have shape [N, tokens]")

    ids = payload.get("ids")
    if ids is None:
        ids = list(range(latents.shape[0]))
    elif len(ids) != latents.shape[0]:
        raise ValueError(f"{path}: ids must contain one entry per sample")

    return {
        "z": latents,
        "x": source_tokens.float(),
        "p": positions,
        "mask": mask,
        "ids": ids,
        "split": payload.get("split", "unknown"),
        "source_epoch": payload.get("epoch"),
        "source_checkpoint": payload.get("checkpoint"),
        "config": payload.get("config"),
    }


def normalise_indices(ids: Any) -> Any:
    """Keep string IDs intact while retaining tensor compatibility for integers."""

    values = ids.cpu().tolist() if isinstance(ids, torch.Tensor) else list(ids)
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return torch.tensor(values, dtype=torch.long)
    return values


def reconstruct_payload(
    *,
    model: Any,
    payload: dict[str, Any],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    checkpoint_epoch: int,
    checkpoint_path: Path,
    config: dict[str, Any],
    config_source: str,
    source_path: Path,
) -> dict[str, Any]:
    data = validate_latent_payload(payload, source_path)
    sample_count = data["z"].shape[0]
    if sample_count == 0:
        raise ValueError(f"Latent payload is empty: {source_path}")

    print(
        f"Decoding {source_path}: samples={sample_count}, "
        f"z={tuple(data['z'].shape)}"
    )
    reconstructed_tokens, output_positions = decode_latents(
        model,
        data["z"],
        data["p"],
        device=device,
        batch_size=batch_size,
    )
    if reconstructed_tokens.shape[0] != sample_count:
        raise ValueError("Decoder changed the number of samples")

    output_mask = data["mask"]
    if reconstructed_tokens.shape[1] != output_mask.shape[1]:
        output_mask = torch.ones(
            reconstructed_tokens.shape[:2],
            dtype=torch.bool,
        )

    source_flat = flatten_token_grid(
        data["x"],
        data["mask"],
        data["p"],
        tokenizer,
        description="Flattening x",
    )
    reconstructed_flat = flatten_token_grid(
        reconstructed_tokens,
        output_mask,
        output_positions,
        tokenizer,
        description="Flattening x_hat",
    )
    if source_flat.shape != reconstructed_flat.shape:
        raise ValueError(
            f"Flat source and reconstruction shapes differ: {tuple(source_flat.shape)} "
            f"!= {tuple(reconstructed_flat.shape)}"
        )

    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "x": source_flat.cpu(),
        "x_hat": reconstructed_flat.cpu(),
        "indices": normalise_indices(data["ids"]),
        "split": data["split"],
        "epoch": checkpoint_epoch,
        "source_epoch": data["source_epoch"],
        "config": config,
        "config_source": config_source,
        "checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint": data["source_checkpoint"],
        "source_latents": str(source_path.resolve()),
    }


def config_cache_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def write_manifest(
    output_dir: Path,
    *,
    checkpoint: Path,
    checkpoint_epoch: int,
    device: torch.device,
    batch_size: int,
    outputs: list[dict[str, Any]],
) -> None:
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "device": str(device),
        "batch_size": batch_size,
        "outputs": outputs,
    }
    (output_dir / "reconstruction_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    destinations = output_paths(args)
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Reconstruction output(s) already exist: {existing}. "
            "Pass --overwrite to replace them."
        )

    device = resolve_device(args.device)
    checkpoint = load_autoencoder_checkpoint(args.checkpoint, device)
    print(f"Loaded checkpoint epoch {checkpoint.epoch} on {device}")

    loaded_payloads = [
        (path, load_latent_payload(path))
        for path in args.latents
    ]
    tokenizer_cache: dict[str, Any] = {}
    manifest_outputs = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for (source_path, payload), destination in zip(loaded_payloads, destinations):
        config, config_source = choose_data_config(
            explicit_path=args.config,
            checkpoint_config=checkpoint.config,
            payload_config=payload.get("config"),
        )
        cache_key = config_cache_key(config)
        if cache_key not in tokenizer_cache:
            print(f"Building tokenizer for config source {config_source} ...")
            tokenizer_cache[cache_key] = tokenizer_from_dataloaders(
                build_dataloaders(config)
            )
        tokenizer = tokenizer_cache[cache_key]

        result = reconstruct_payload(
            model=checkpoint.model,
            payload=payload,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            checkpoint_epoch=checkpoint.epoch,
            checkpoint_path=checkpoint.path,
            config=config,
            config_source=config_source,
            source_path=source_path,
        )
        torch.save(result, destination)
        manifest_outputs.append(
            {
                "source": str(source_path.resolve()),
                "output": str(destination.resolve()),
                "samples": result["x_hat"].shape[0],
                "weight_width": result["x_hat"].shape[1],
                "split": result["split"],
                "config_source": config_source,
            }
        )
        print(
            f"Saved {destination}: x={tuple(result['x'].shape)}, "
            f"x_hat={tuple(result['x_hat'].shape)}"
        )

    write_manifest(
        args.output_dir,
        checkpoint=checkpoint.path,
        checkpoint_epoch=checkpoint.epoch,
        device=device,
        batch_size=args.batch_size,
        outputs=manifest_outputs,
    )
    print(f"Done. Reconstructed {len(manifest_outputs)} payload(s).")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

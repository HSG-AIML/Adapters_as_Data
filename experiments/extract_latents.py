"""Encode SANE dataset splits into reusable latent artifact payloads.

This is the first half of the post-training artifact bridge:

    SANE checkpoint + configured LoRA population
      -> encoder
      -> latents_<split>.pt

Each output payload contains the encoded latent ``z`` together with the original
token grid ``x``, positions ``p``, mask, sample IDs, checkpoint metadata, and the
resolved config required for later reconstruction.

Example:

    uv run python experiments/extract_latents.py \
        path/to/checkpoint.pt \
        --output-dir outputs/latents \
        --splits train val test

The checkpoint normally contains the complete resolved experiment config. Use
``--config`` only when dataset paths or other data settings must be overridden.
Torch payloads are pickle-backed and must come from trusted sources.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from sane.postprocessing import (
    build_dataloaders,
    choose_data_config,
    load_autoencoder_checkpoint,
    resolve_device,
)


FORMAT_NAME = "sane.latents"
FORMAT_VERSION = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Trusted SANE checkpoint.pt file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML/JSON data-config override.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
        help="Dataset splits to encode (default: train val test).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. cuda:0 or cpu (default: auto).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing latent payloads for the selected splits.",
    )
    return parser.parse_args(argv)


def _batch_parts(batch: Any) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, Any]:
    mask = None
    ids = None
    if len(batch) == 2:
        tokens, positions = batch
    elif len(batch) == 3:
        tokens, mask, positions = batch
    elif len(batch) == 4:
        tokens, mask, positions, ids = batch
    else:
        raise ValueError(
            f"Unexpected batch length {len(batch)}; expected (x,p), (x,mask,p), "
            "or (x,mask,p,ids)"
        )
    return tokens, mask, positions, ids


@torch.inference_mode()
def extract_split(
    model: Any,
    dataloader: Any,
    *,
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    """Run the encoder over one split without changing sample order."""

    latent_chunks: list[torch.Tensor] = []
    token_chunks: list[torch.Tensor] = []
    position_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    sample_ids: list[Any] = []
    mask_mode: bool | None = None
    id_mode: bool | None = None
    sample_offset = 0

    for batch in tqdm(dataloader, desc=f"Extracting {split_name}"):
        tokens, mask, positions, ids = _batch_parts(batch)
        has_mask = mask is not None
        has_ids = ids is not None
        if mask_mode is not None and mask_mode != has_mask:
            raise ValueError(f"Split {split_name!r} mixes batches with and without masks")
        if id_mode is not None and id_mode != has_ids:
            raise ValueError(f"Split {split_name!r} mixes batches with and without IDs")
        mask_mode = has_mask
        id_mode = has_ids

        token_device = tokens.to(device)
        position_device = positions.to(device)
        latents = model.encoder(token_device, position_device)

        latent_chunks.append(latents.cpu())
        token_chunks.append(tokens.cpu())
        position_chunks.append(positions.cpu())
        if mask is not None:
            mask_chunks.append(mask.cpu())

        batch_size = tokens.shape[0]
        if ids is None:
            sample_ids.extend(range(sample_offset, sample_offset + batch_size))
        elif isinstance(ids, torch.Tensor):
            sample_ids.extend(ids.cpu().tolist())
        else:
            sample_ids.extend(list(ids))
        sample_offset += batch_size

    if not latent_chunks:
        raise ValueError(f"Dataset split {split_name!r} is empty")

    result = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "z": torch.cat(latent_chunks),
        "x": torch.cat(token_chunks),
        "p": torch.cat(position_chunks),
        "mask": torch.cat(mask_chunks) if mask_chunks else None,
        "ids": sample_ids,
        "split": split_name,
    }
    print(
        f"[{split_name}] samples={result['z'].shape[0]}, "
        f"z={tuple(result['z'].shape)}, x={tuple(result['x'].shape)}"
    )
    return result


def write_manifest(
    output_dir: Path,
    *,
    checkpoint: Path,
    checkpoint_epoch: int,
    config_source: str,
    device: torch.device,
    outputs: dict[str, dict[str, Any]],
) -> None:
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "config_source": config_source,
        "device": str(device),
        "splits": {
            split: {
                "path": str(details["path"].resolve()),
                "samples": details["samples"],
                "z_shape": details["z_shape"],
                "x_shape": details["x_shape"],
            }
            for split, details in outputs.items()
        },
    }
    (output_dir / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    selected_splits = list(dict.fromkeys(args.splits))
    output_paths = {
        split: args.output_dir / f"latents_{split}.pt"
        for split in selected_splits
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Latent output(s) already exist: {existing}. Pass --overwrite to replace them."
        )

    device = resolve_device(args.device)
    checkpoint = load_autoencoder_checkpoint(args.checkpoint, device)
    config, config_source = choose_data_config(
        explicit_path=args.config,
        checkpoint_config=checkpoint.config,
    )
    print(
        f"Loaded checkpoint epoch {checkpoint.epoch} on {device}; "
        f"config={config_source}"
    )
    train_loader, validation_loader, test_loader = build_dataloaders(config)
    loaders = {
        "train": train_loader,
        "val": validation_loader,
        "test": test_loader,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for split_name in selected_splits:
        dataloader = loaders[split_name]
        if dataloader is None:
            print(f"[{split_name}] split is not configured; skipping")
            continue

        payload = extract_split(
            checkpoint.model,
            dataloader,
            device=device,
            split_name=split_name,
        )
        payload.update(
            {
                "epoch": checkpoint.epoch,
                "config": config,
                "config_source": config_source,
                "checkpoint": str(checkpoint.path.resolve()),
            }
        )
        output_path = output_paths[split_name]
        torch.save(payload, output_path)
        outputs[split_name] = {
            "path": output_path,
            "samples": payload["z"].shape[0],
            "z_shape": list(payload["z"].shape),
            "x_shape": list(payload["x"].shape),
        }
        print(f"[{split_name}] saved {output_path}")

    if not outputs:
        raise ValueError("None of the selected dataset splits were available")
    write_manifest(
        args.output_dir,
        checkpoint=checkpoint.path,
        checkpoint_epoch=checkpoint.epoch,
        config_source=config_source,
        device=device,
        outputs=outputs,
    )
    print(f"Done. Extracted {len(outputs)} split(s) into {args.output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

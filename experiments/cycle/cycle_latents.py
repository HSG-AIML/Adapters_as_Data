"""Run repeated SANE encode/decode cycles over a fixed set of LoRA adapters.

For an original tokenized adapter ``x_0``, this script computes

    x_0 -> E -> z_1 -> D -> x_1 -> E -> z_2 -> D -> ... -> x_R

without changing the trained autoencoder between rounds. The procedure is the
same reconstruction cycle used in the paper; this script only makes its inputs,
outputs, and provenance explicit.

The starting token grids can come from either an extraction payload or a data
split. Extraction payloads are usually simpler and avoid reading every adapter
from the source dataset again, although the tokenizer still needs the dataset
configuration and reference weights.

Example using extracted validation latents:

    uv run python experiments/cycle/cycle_latents.py \
        path/to/checkpoint.pt \
        --latents outputs/latents/latents_val.pt \
        --output-dir outputs/cycles/validation \
        --num-rounds 10 --rows 0 1 2 3

Example using a configured dataset split:

    uv run python experiments/cycle/cycle_latents.py \
        path/to/checkpoint.pt \
        --config path/to/trial.yaml \
        --split val \
        --output-dir outputs/cycles/validation \
        --num-rounds 10 --limit 4

All ``.pt`` inputs are pickle-backed and must come from trusted sources.

Outputs:

- ``round_XX/x_and_x_hat.pt`` stores the original flat weights in ``x`` and
  round ``XX`` in ``x_hat``. Round zero is the identity baseline.
- ``cycle_summary.pt`` stores the complete flat-weight and latent trajectories.
- ``cycle_config.json`` records the invocation and source artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from sane.model.autoencoder import SANEAutoEncoder
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--latents",
        type=Path,
        help="Trusted extract_latents.py payload supplying x_0, p, mask, and IDs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "YAML/JSON data config override. By default, use the latent "
            "payload's embedded config and then the checkpoint's."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split used when --latents is omitted (default: val).",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--rows",
        nargs="+",
        type=int,
        help="Zero-based source rows to cycle (default: all).",
    )
    selection.add_argument(
        "--limit",
        type=int,
        help="Cycle only the first N source rows.",
    )
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. cuda:0 or cpu (default: auto).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing cycle payloads in the output directory.",
    )
    return parser.parse_args(argv)


def load_source_from_latents(payload: dict[str, Any]) -> dict[str, Any]:
    required = ("x", "p")
    missing = [field for field in required if field not in payload]
    if missing:
        raise KeyError(f"Latent payload is missing required fields: {missing}")

    x = payload["x"]
    positions = payload["p"]
    mask = payload.get("mask")
    if not isinstance(x, torch.Tensor) or x.ndim != 3:
        raise ValueError("Latent payload x must have shape [N, tokens, features]")
    if not isinstance(positions, torch.Tensor) or positions.shape[:2] != x.shape[:2]:
        raise ValueError("Latent payload p must align with x on samples and tokens")
    if mask is not None and (
        not isinstance(mask, torch.Tensor) or mask.shape != x.shape[:2]
    ):
        raise ValueError("Latent payload mask must have shape [N, tokens]")

    ids = payload.get("ids")
    if ids is not None and len(ids) != x.shape[0]:
        raise ValueError("Latent payload ids must have one entry per sample")
    return {
        "x": x.float(),
        "p": positions,
        "mask": mask,
        "ids": ids,
        "split": payload.get("split", "unknown"),
        "source_epoch": payload.get("epoch"),
    }


@torch.no_grad()
def load_source_from_split(loader: Any, split_name: str) -> dict[str, Any]:
    if loader is None:
        raise ValueError(f"The configured {split_name!r} split is unavailable")

    all_x: list[torch.Tensor] = []
    all_positions: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_ids: list[Any] = []
    for batch in tqdm(loader, desc=f"Reading {split_name}"):
        mask = None
        ids = None
        if len(batch) == 2:
            x, positions = batch
        elif len(batch) == 3:
            x, mask, positions = batch
        elif len(batch) == 4:
            x, mask, positions, ids = batch
        else:
            raise ValueError(f"Unexpected dataset batch length: {len(batch)}")

        all_x.append(x.cpu())
        all_positions.append(positions.cpu())
        if mask is not None:
            all_masks.append(mask.cpu())
        if ids is not None:
            all_ids.extend(ids.tolist() if isinstance(ids, torch.Tensor) else list(ids))

    if not all_x:
        raise ValueError(f"The configured {split_name!r} split is empty")
    return {
        "x": torch.cat(all_x).float(),
        "p": torch.cat(all_positions),
        "mask": torch.cat(all_masks) if all_masks else None,
        "ids": all_ids or None,
        "split": split_name,
        "source_epoch": None,
    }


def select_source_rows(
    source: dict[str, Any],
    rows: Sequence[int] | None,
    limit: int | None,
) -> tuple[dict[str, Any], list[int]]:
    sample_count = source["x"].shape[0]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        selected = list(range(min(limit, sample_count)))
    elif rows is not None:
        selected = list(rows)
    else:
        selected = list(range(sample_count))

    invalid = [row for row in selected if row < 0 or row >= sample_count]
    if invalid:
        raise IndexError(
            f"Source rows {invalid} are outside the 0..{sample_count - 1} range"
        )
    if not selected:
        raise ValueError("No source rows were selected")
    if len(set(selected)) != len(selected):
        raise ValueError("Selected rows contain duplicates")

    index = torch.tensor(selected, dtype=torch.long)
    result = dict(source)
    for field in ("x", "p", "mask"):
        if result[field] is not None:
            result[field] = result[field].index_select(0, index)
    if result["ids"] is not None:
        result["ids"] = [result["ids"][row] for row in selected]
    return result, selected


@torch.no_grad()
def encode_all(
    model: SANEAutoEncoder,
    token_grid: torch.Tensor,
    positions: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in tqdm(
        range(0, token_grid.shape[0], batch_size),
        desc="Encoding",
        leave=False,
    ):
        x_batch = token_grid[start : start + batch_size].to(device)
        p_batch = positions[start : start + batch_size].to(device)
        chunks.append(model.encoder(x_batch, p_batch).cpu())
    return torch.cat(chunks)


def save_round(
    output_dir: Path,
    round_number: int,
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    indices: Any,
    split: str,
    checkpoint_epoch: int,
    source_epoch: Any,
    config: dict[str, Any],
    checkpoint: Path,
) -> Path:
    round_dir = output_dir / f"round_{round_number:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    path = round_dir / "x_and_x_hat.pt"
    torch.save(
        {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "x": original.cpu(),
            "x_hat": reconstruction.cpu(),
            "indices": indices,
            "split": split,
            "epoch": checkpoint_epoch,
            "source_epoch": source_epoch,
            "round": round_number,
            "config": config,
            "checkpoint": str(checkpoint.resolve()),
        },
        path,
    )
    print(f"Saved round {round_number:2d}: {path}")
    return path


def write_manifest(
    args: argparse.Namespace,
    output_path: Path,
    config_source: str,
    selected_rows: list[int],
    device: torch.device,
) -> None:
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "latents": str(args.latents.resolve()) if args.latents else None,
        "config_source": config_source,
        "split": args.split if args.latents is None else None,
        "selected_source_rows": selected_rows,
        "num_rounds": args.num_rounds,
        "batch_size": args.batch_size,
        "device": str(device),
    }
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.num_rounds < 1:
        raise ValueError("--num-rounds must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.latents is not None and not args.latents.is_file():
        raise FileNotFoundError(f"Latent payload not found: {args.latents}")
    if args.config is not None and not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")

    existing_outputs = list(args.output_dir.glob("round_*/x_and_x_hat.pt"))
    summary_path = args.output_dir / "cycle_summary.pt"
    if summary_path.exists():
        existing_outputs.append(summary_path)
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"Cycle outputs already exist under {args.output_dir}. Pass --overwrite "
            "to replace the payloads."
        )
    if args.overwrite:
        for path in existing_outputs:
            path.unlink()

    device = resolve_device(args.device)
    latent_payload = None
    if args.latents is not None:
        latent_payload = torch.load(
            args.latents,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(latent_payload, Mapping):
            raise ValueError("Latent payload must contain a mapping")
        latent_payload = dict(latent_payload)

    checkpoint = load_autoencoder_checkpoint(args.checkpoint, device)
    model = checkpoint.model
    checkpoint_epoch = checkpoint.epoch
    config, config_source = choose_data_config(
        explicit_path=args.config,
        checkpoint_config=checkpoint.config,
        payload_config=latent_payload.get("config") if latent_payload else None,
    )
    print(
        f"Checkpoint epoch {checkpoint_epoch}; device={device}; "
        f"config={config_source}"
    )

    print("Building tokenizer and data pipeline ...")
    dataloaders = build_dataloaders(config)
    train_loader, validation_loader, test_loader = dataloaders
    tokenizer = tokenizer_from_dataloaders(dataloaders)
    if latent_payload is not None:
        source = load_source_from_latents(latent_payload)
    else:
        loaders = {
            "train": train_loader,
            "val": validation_loader,
            "test": test_loader,
        }
        source = load_source_from_split(loaders[args.split], args.split)
    source, selected_rows = select_source_rows(source, args.rows, args.limit)

    original_grid = source["x"]
    positions = source["p"]
    mask = source["mask"]
    sample_count, token_count = original_grid.shape[:2]
    if mask is None:
        mask = torch.ones(sample_count, token_count, dtype=torch.bool)

    indices = source["ids"]
    if indices is None:
        indices = torch.tensor(selected_rows, dtype=torch.long)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        args,
        args.output_dir / "cycle_config.json",
        config_source,
        selected_rows,
        device,
    )

    print(
        f"Cycling {sample_count} adapter(s), token grid={tuple(original_grid.shape)}, "
        f"rounds={args.num_rounds}"
    )
    original_flat = flatten_token_grid(
        original_grid,
        mask,
        positions,
        tokenizer,
        description="Flattening x_0",
    )
    print(f"Flat adapter width: {original_flat.shape[1]:,}")

    flat_trajectory = [original_flat]
    latent_trajectory = []
    save_round(
        args.output_dir,
        0,
        original_flat,
        original_flat,
        indices,
        source["split"],
        checkpoint_epoch,
        source["source_epoch"],
        config,
        args.checkpoint,
    )

    current_grid = original_grid
    current_positions = positions
    current_mask = mask
    for round_number in range(1, args.num_rounds + 1):
        print(f"\nRound {round_number}/{args.num_rounds}")
        latents = encode_all(
            model,
            current_grid,
            current_positions,
            device,
            args.batch_size,
        )
        reconstruction_grid, output_positions = decode_latents(
            model,
            latents,
            current_positions,
            device=device,
            batch_size=args.batch_size,
        )
        if reconstruction_grid.shape[1] != current_mask.shape[1]:
            current_mask = torch.ones(
                sample_count,
                reconstruction_grid.shape[1],
                dtype=torch.bool,
            )

        reconstruction_flat = flatten_token_grid(
            reconstruction_grid,
            current_mask,
            output_positions,
            tokenizer,
            description=f"Flattening round {round_number}",
        )
        if reconstruction_flat.shape != original_flat.shape:
            raise ValueError(
                f"Round {round_number} reconstructed shape "
                f"{tuple(reconstruction_flat.shape)}, expected {tuple(original_flat.shape)}"
            )

        latent_trajectory.append(latents)
        flat_trajectory.append(reconstruction_flat)
        save_round(
            args.output_dir,
            round_number,
            original_flat,
            reconstruction_flat,
            indices,
            source["split"],
            checkpoint_epoch,
            source["source_epoch"],
            config,
            args.checkpoint,
        )

        mse_original = torch.mean((reconstruction_flat - original_flat) ** 2).item()
        mse_previous = torch.mean(
            (reconstruction_flat - flat_trajectory[-2]) ** 2
        ).item()
        print(f"MSE to original={mse_original:.6e}; to previous={mse_previous:.6e}")
        current_grid = reconstruction_grid
        current_positions = output_positions

    flat_stack = torch.stack(flat_trajectory)
    latent_stack = torch.stack(latent_trajectory)
    mse_to_original = torch.stack(
        [torch.mean((weights - flat_stack[0]) ** 2) for weights in flat_stack]
    )
    mse_step = torch.stack(
        [
            torch.tensor(0.0)
            if round_number == 0
            else torch.mean(
                (flat_stack[round_number] - flat_stack[round_number - 1]) ** 2
            )
            for round_number in range(flat_stack.shape[0])
        ]
    )

    torch.save(
        {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "x_flat": flat_stack,
            "z": latent_stack,
            "mse_to_orig": mse_to_original,
            "mse_step": mse_step,
            "indices": indices,
            "split": source["split"],
            "epoch": checkpoint_epoch,
            "source_epoch": source["source_epoch"],
            "num_rounds": args.num_rounds,
            "selected_source_rows": selected_rows,
            "config": config,
            "checkpoint": str(args.checkpoint.resolve()),
        },
        summary_path,
    )

    print("\nCycle MSE summary:")
    for round_number in range(flat_stack.shape[0]):
        print(
            f"  round {round_number:2d}: to original="
            f"{mse_to_original[round_number].item():.6e}; step="
            f"{mse_step[round_number].item():.6e}"
        )
    print(f"Summary: {summary_path}")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

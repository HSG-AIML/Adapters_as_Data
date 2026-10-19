"""Linearly interpolate between two token-shaped SANE latents.

The sweep is evenly spaced and includes both endpoints: ``alpha = 0`` is sample
A, ``alpha = 1`` is sample B.

The result is written as a multi-row latent payload for
``experiments/reconstruct_latents.py``. Sample A supplies ``x``, ``p``, and
``mask`` reconstruction metadata, but its ``x`` is not ground truth for the
interpolated latents.

Example:

    uv run python experiments/interpolation/interpolate_latents.py \
        outputs/latents/latents_val.pt \
        --output outputs/interpolation/latents_A2_B13.pt \
        --sample-a 2 --sample-b 13 --steps 10

Torch payloads are pickle-backed and must come from trusted sources.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from interpolation import (
    DEFAULT_STEPS,
    interpolate,
    interpolation_alphas,
    pair_geometry,
    sample_ids,
    tensor_stats,
    tile_rows,
    validate_sample_pair,
)


FORMAT_NAME = "sane.latents"
FORMAT_VERSION = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("latents", type=Path, help="Trusted latent population payload.")
    parser.add_argument("--output", type=Path, required=True, help="Interpolated .pt file.")
    parser.add_argument("--sample-a", type=int, default=2)
    parser.add_argument("--sample-b", type=int, default=13)
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Interpolation points including both endpoints (default: {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing interpolation payload.",
    )
    return parser.parse_args(argv)


def load_latent_population(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Latent payload not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Latent payload must contain a mapping: {path}")
    payload = dict(payload)

    missing = [field for field in ("z", "x", "p") if field not in payload]
    if missing:
        raise KeyError(f"Latent payload is missing required fields: {missing}")
    latents = payload["z"]
    source_tokens = payload["x"]
    positions = payload["p"]
    for field, tensor in (
        ("z", latents),
        ("x", source_tokens),
        ("p", positions),
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"payload[{field!r}] must be a rank-3 tensor")
    if latents.shape[:2] != source_tokens.shape[:2]:
        raise ValueError("z and x must align on samples and tokens")
    if positions.shape[:2] != latents.shape[:2] or positions.shape[2] != 3:
        raise ValueError("p must have shape [N, tokens, 3] aligned with z")

    mask = payload.get("mask")
    if mask is not None and (
        not isinstance(mask, torch.Tensor) or mask.shape != latents.shape[:2]
    ):
        raise ValueError("mask must have shape [N, tokens]")
    sample_ids(payload, latents.shape[0])
    return payload


def build_result_payload(
    source: dict[str, Any],
    interpolated: torch.Tensor,
    alphas: Sequence[float],
    *,
    sample_a: int,
    sample_b: int,
    id_a: Any,
    id_b: Any,
    l2_distance: float,
    cosine: float,
    source_path: Path,
) -> dict[str, Any]:
    step_count = interpolated.shape[0]
    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "z": interpolated.to(source["z"].dtype),
        "ids": [
            f"interp_{index}_alpha{alpha:.3f}"
            for index, alpha in enumerate(alphas)
        ],
        "split": f"interpolation_A{sample_a}_B{sample_b}",
        "epoch": source.get("epoch"),
        "config": source.get("config"),
        "checkpoint": source.get("checkpoint"),
        "source_file": str(source_path.resolve()),
        "space": "latent",
        "interp_sample_a": sample_a,
        "interp_sample_b": sample_b,
        "interp_id_a": id_a,
        "interp_id_b": id_b,
        "interp_alphas": list(alphas),
        "interp_n_steps": step_count,
        "interp_l2_dist": l2_distance,
        "interp_cos_sim": cosine,
        "reference_note": (
            "x, p, and mask are tiled from sample A and are reconstruction "
            "metadata, not ground truth for the interpolated latents"
        ),
    }
    # Positions are identical across samples (same tokenization scheme), so
    # tiling sample A is correct; x is only the reference token grid that
    # reconstruction needs for positional ordering.
    for field in ("x", "p", "mask"):
        value = source.get(field)
        payload[field] = (
            tile_rows(value[sample_a], step_count).to(value.dtype)
            if isinstance(value, torch.Tensor)
            else None
        )
    return payload


def write_manifest(
    args: argparse.Namespace,
    *,
    source_shape: tuple[int, ...],
    result_shape: tuple[int, ...],
    alphas: Sequence[float],
    id_a: Any,
    id_b: Any,
    l2_distance: float,
    cosine: float,
) -> Path:
    manifest = {
        "space": "latent",
        "source": str(args.latents.resolve()),
        "source_shape": list(source_shape),
        "output": str(args.output.resolve()),
        "result_shape": list(result_shape),
        "sample_a": args.sample_a,
        "sample_b": args.sample_b,
        "id_a": id_a,
        "id_b": id_b,
        "steps": args.steps,
        "alphas": list(alphas),
        "l2_distance": l2_distance,
        "cosine_similarity": cosine,
    }
    path = args.output.with_name(f"{args.output.stem}_manifest.json")
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> None:
    alphas = interpolation_alphas(args.steps)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Interpolation output already exists: {args.output}. "
            "Pass --overwrite to replace it."
        )

    payload = load_latent_population(args.latents)
    latents = payload["z"]
    validate_sample_pair(args.sample_a, args.sample_b, latents.shape[0])
    ids = sample_ids(payload, latents.shape[0])
    id_a = ids[args.sample_a]
    id_b = ids[args.sample_b]

    sample_a = latents[args.sample_a].float()
    sample_b = latents[args.sample_b].float()
    l2_distance, cosine = pair_geometry(sample_a, sample_b)
    print(f"Loaded {args.latents}: z={tuple(latents.shape)}")
    print(f"A row={args.sample_a}, id={id_a}: {tensor_stats(sample_a)}")
    print(f"B row={args.sample_b}, id={id_b}: {tensor_stats(sample_b)}")
    print(f"A to B: L2={l2_distance:.6f}, cosine={cosine:.6f}")

    interpolated = interpolate(sample_a, sample_b, alphas)
    print(f"Interpolated {args.steps} steps:")
    for index, alpha in enumerate(alphas):
        step = interpolated[index]
        print(
            f"  step {index:2d}  alpha={alpha:.3f}  "
            f"mean={step.mean():.6f}  std={step.std():.6f}"
        )

    result_payload = build_result_payload(
        payload,
        interpolated,
        alphas,
        sample_a=args.sample_a,
        sample_b=args.sample_b,
        id_a=id_a,
        id_b=id_b,
        l2_distance=l2_distance,
        cosine=cosine,
        source_path=args.latents,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result_payload, args.output)

    manifest_path = write_manifest(
        args,
        source_shape=tuple(latents.shape),
        result_shape=tuple(interpolated.shape),
        alphas=alphas,
        id_a=id_a,
        id_b=id_b,
        l2_distance=l2_distance,
        cosine=cosine,
    )
    print(f"Saved latent interpolation payload: {args.output}")
    print(f"Manifest: {manifest_path}")
    print("Next: reconstruct the payload with experiments/reconstruct_latents.py")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

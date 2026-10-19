"""Linearly interpolate between two flat LoRA weight vectors.

This is the matched raw-weight control for ``interpolate_latents.py``. It uses
the same evenly spaced sweep, but bypasses SANE reconstruction:

    flat LoRA A/B -> interpolation -> x_hat -> image renderer

For a matched comparison, use the ``x`` field from the reconstruction of the
same latent population and select the same ``--sample-a`` and ``--sample-b``
rows used by the latent script.

Example:

    uv run python experiments/interpolation/interpolate_weights.py \
        outputs/reconstructions/x_and_x_hat_val.pt \
        --field x \
        --output outputs/interpolation/weights_A2_B13.pt \
        --sample-a 2 --sample-b 13 --steps 10

The result is a flat-weight payload for ``experiments/generate_images.py``. Its
``x`` field is sample A as a reference, not ground truth for the interpolated
``x_hat``; render with ``--fields x_hat --no-scores``.

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


FORMAT_NAME = "sane.reconstructed_weights"
FORMAT_VERSION = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "weights",
        type=Path,
        help="Trusted tensor or payload containing a flat weight population.",
    )
    parser.add_argument(
        "--field",
        default="x",
        help="Weight field when the input is a dictionary (default: x).",
    )
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


def load_weight_population(
    path: Path,
    field: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Weight payload not found: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, torch.Tensor):
        weights = loaded
        metadata: dict[str, Any] = {}
    elif isinstance(loaded, Mapping):
        if field not in loaded:
            raise KeyError(
                f"Input payload has no {field!r} field; available fields: "
                f"{sorted(str(key) for key in loaded)}"
            )
        weights = loaded[field]
        metadata = dict(loaded)
    else:
        raise ValueError("Weight input must be a tensor or dictionary")

    if not isinstance(weights, torch.Tensor) or weights.ndim < 2:
        raise ValueError(f"Weight field {field!r} must have a sample axis")
    return weights.reshape(weights.shape[0], -1), metadata


def build_result_payload(
    source: dict[str, Any],
    sample_a_weights: torch.Tensor,
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
    source_field: str,
) -> dict[str, Any]:
    step_count = interpolated.shape[0]
    ids = [
        f"interp_{index}_alpha{alpha:.3f}"
        for index, alpha in enumerate(alphas)
    ]
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "x": tile_rows(sample_a_weights, step_count),
        "x_hat": interpolated.to(sample_a_weights.dtype),
        "indices": ids,
        "ids": ids,
        "split": f"weight_interpolation_A{sample_a}_B{sample_b}",
        "epoch": source.get("epoch"),
        "config": source.get("config"),
        "checkpoint": source.get("checkpoint"),
        "source_file": str(source_path.resolve()),
        "source_field": source_field,
        "space": "weight",
        "interp_sample_a": sample_a,
        "interp_sample_b": sample_b,
        "interp_id_a": id_a,
        "interp_id_b": id_b,
        "interp_alphas": list(alphas),
        "interp_n_steps": step_count,
        "interp_l2_dist": l2_distance,
        "interp_cos_sim": cosine,
        "reference_note": (
            "x contains sample A as a reference and is not ground truth for x_hat"
        ),
    }


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
        "space": "weight",
        "source": str(args.weights.resolve()),
        "source_field": args.field,
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

    weights, metadata = load_weight_population(args.weights, args.field)
    validate_sample_pair(args.sample_a, args.sample_b, weights.shape[0])
    ids = sample_ids(metadata, weights.shape[0])
    id_a = ids[args.sample_a]
    id_b = ids[args.sample_b]

    sample_a = weights[args.sample_a].float()
    sample_b = weights[args.sample_b].float()
    l2_distance, cosine = pair_geometry(sample_a, sample_b)
    print(f"Loaded {args.weights}: weights={tuple(weights.shape)}")
    print(f"A row={args.sample_a}, id={id_a}: {tensor_stats(sample_a)}")
    print(f"B row={args.sample_b}, id={id_b}: {tensor_stats(sample_b)}")
    print(f"A to B: L2={l2_distance:.4f}, cosine={cosine:.6f}")

    interpolated = interpolate(sample_a, sample_b, alphas)
    print(f"Interpolated {args.steps} steps:")
    for index, alpha in enumerate(alphas):
        step = interpolated[index]
        print(
            f"  step {index:2d}  alpha={alpha:.3f}  "
            f"mean={step.mean():.6f}  std={step.std():.6f}"
        )

    result_payload = build_result_payload(
        metadata,
        weights[args.sample_a],
        interpolated,
        alphas,
        sample_a=args.sample_a,
        sample_b=args.sample_b,
        id_a=id_a,
        id_b=id_b,
        l2_distance=l2_distance,
        cosine=cosine,
        source_path=args.weights,
        source_field=args.field,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result_payload, args.output)

    manifest_path = write_manifest(
        args,
        source_shape=tuple(weights.shape),
        result_shape=tuple(interpolated.shape),
        alphas=alphas,
        id_a=id_a,
        id_b=id_b,
        l2_distance=l2_distance,
        cosine=cosine,
    )
    print(f"Saved weight interpolation payload: {args.output}")
    print(f"Manifest: {manifest_path}")
    print("Next: render x_hat directly with experiments/generate_images.py")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

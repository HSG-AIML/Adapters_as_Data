"""Perturb flat LoRA adapters with Gaussian noise.

This is the matched raw-weight control for ``noise_latents.py``. It uses the
same noise model, but bypasses SANE entirely:

    theta' = theta + a * sigma * N(0, 1)

``sigma`` is the population standard deviation of the adapter weights, so the
``a`` axis means the same thing in both spaces; ``a = 0`` keeps the adapter
unperturbed. Pass ``--sigma`` to reuse a previously measured population spread.

One row is written per selected row and noise scale, in row-major order, as a
flat-weight payload for ``experiments/generate_images.py``. Its ``x`` field is
the unperturbed adapter, so ``--fields x x_hat`` renders both; ``--fields x_hat
--no-scores`` renders the perturbed adapters alone.

Example:

    uv run python experiments/perturbation/noise_weights.py \
        outputs/reconstructions/x_and_x_hat_val.pt \
        --field x \
        --output outputs/perturbation/weights_noise.pt \
        --rows 0 1 2 3 \
        --noise-scales 0 0.5 0.75 1.0

Torch payloads are pickle-backed and must come from trusted sources.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from perturbation import (
    DEFAULT_SEED,
    add_gaussian_noise,
    print_statistics,
    summarize,
)


FORMAT_NAME = "sane.reconstructed_weights"
FORMAT_VERSION = 1
DEFAULT_NOISE_SCALES = (0.0, 0.5, 0.75, 1.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "weights",
        type=Path,
        help="Trusted tensor or payload containing a flat adapter population.",
    )
    parser.add_argument(
        "--field",
        default="x",
        help="Weight field when the input is a dictionary (default: x).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Perturbed .pt file.")
    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        required=True,
        help="Adapter rows to perturb.",
    )
    parser.add_argument(
        "--noise-scales",
        nargs="+",
        type=float,
        default=DEFAULT_NOISE_SCALES,
        help="Noise scales in units of sigma (default: 0 0.5 0.75 1.0).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        help="Override the population spread instead of measuring it.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for the noise draws (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--mmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Memory-map tensors from the Torch archive when possible (default: true).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing perturbation payload.",
    )
    return parser.parse_args(argv)


def load_weight_population(
    path: Path,
    *,
    field: str,
    mmap: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Weight payload not found: {path}")
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    except (TypeError, RuntimeError, ValueError) as error:
        if not mmap:
            raise
        print(f"  mmap unavailable ({type(error).__name__}); loading fully")
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


def source_ids(metadata: dict[str, Any], sample_count: int) -> list[Any]:
    values = metadata.get("indices", metadata.get("ids"))
    if values is None:
        return list(range(sample_count))
    if len(values) != sample_count:
        raise ValueError(
            f"Input IDs contain {len(values)} entries for {sample_count} adapters"
        )
    if isinstance(values, torch.Tensor):
        return values.cpu().tolist()
    return list(values)


def build_result_payload(
    metadata: dict[str, Any],
    original: torch.Tensor,
    perturbed: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    sigma: float,
    population_shape: tuple[int, ...],
) -> dict[str, Any]:
    ids = [row["id"] for row in rows]
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "x": original,
        "x_hat": perturbed,
        "indices": ids,
        "ids": ids,
        "split": "weight_noise",
        "epoch": metadata.get("epoch"),
        "config": metadata.get("config"),
        "checkpoint": metadata.get("checkpoint"),
        "source_file": str(args.weights.resolve()),
        "source_field": args.field,
        "source_shape": list(population_shape),
        "space": "weight",
        "noise_scales": list(args.noise_scales),
        "noise_sigma": sigma,
        "noise_seed": args.seed,
        "noise_rows": list(args.rows),
        "noise_perturbations": rows,
        "reference_note": (
            "x contains the unperturbed adapter for each row; a = 0 rows have "
            "x_hat equal to x"
        ),
    }


def write_manifest(
    args: argparse.Namespace,
    *,
    population_shape: tuple[int, ...],
    sigma: float,
    ids: Sequence[Any],
    rows: list[dict[str, Any]],
) -> Path:
    manifest = {
        "space": "weight",
        "source": str(args.weights.resolve()),
        "source_field": args.field,
        "source_shape": list(population_shape),
        "output": str(args.output.resolve()),
        "rows": list(args.rows),
        "row_ids": [ids[row] for row in args.rows],
        "noise_scales": list(args.noise_scales),
        "sigma": sigma,
        "sigma_override": args.sigma,
        "seed": args.seed,
        "perturbations": rows,
    }
    path = args.output.with_name(f"{args.output.stem}_manifest.json")
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> None:
    scales = list(dict.fromkeys(args.noise_scales))
    if not scales:
        raise ValueError("At least one --noise-scales value is required")
    if any(scale < 0 for scale in scales):
        raise ValueError("--noise-scales must be non-negative")
    if args.sigma is not None and args.sigma < 0:
        raise ValueError("--sigma must be non-negative")
    selected_rows = list(dict.fromkeys(args.rows))
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Perturbation output already exists: {args.output}. "
            "Pass --overwrite to replace it."
        )

    print(f"Loading {args.weights}")
    weights, metadata = load_weight_population(
        args.weights,
        field=args.field,
        mmap=args.mmap,
    )
    sample_count, width = weights.shape
    print(f"  population: {tuple(weights.shape)}")
    invalid = [row for row in selected_rows if row < 0 or row >= sample_count]
    if invalid:
        raise IndexError(f"--rows {invalid} are outside 0..{sample_count - 1}")
    ids = source_ids(metadata, sample_count)

    population_stats = summarize(weights)
    print_statistics(population_stats, "population")
    sigma = population_stats["std"] if args.sigma is None else args.sigma
    print(f"  sigma used: {sigma:.6f}")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    original_rows = []
    perturbed_rows = []
    rows: list[dict[str, Any]] = []

    for sample in selected_rows:
        adapter = weights[sample].float().clone()
        print(f"\nSample row {sample}, id={ids[sample]}")
        for scale in scales:
            perturbed, noise_std = add_gaussian_noise(
                adapter,
                scale=scale,
                sigma=sigma,
                generator=generator,
            )
            displacement = float((perturbed - adapter).norm().item())
            original_rows.append(adapter)
            perturbed_rows.append(perturbed)
            rows.append(
                {
                    "id": f"noise_sample{sample}_a{scale:.3f}",
                    "sample_index": sample,
                    "sample_id": ids[sample],
                    "noise_scale": scale,
                    "noise_std": noise_std,
                    "displacement_norm": displacement,
                }
            )
            print(
                f"  a={scale:<6.3f} noise std={noise_std:.6f}  "
                f"||dtheta||={displacement:.4f}  "
                f"mean={perturbed.mean():.6f}  std={perturbed.std():.6f}"
            )

    original = torch.stack(original_rows, dim=0).to(weights.dtype)
    perturbed = torch.stack(perturbed_rows, dim=0).to(weights.dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        build_result_payload(
            metadata,
            original,
            perturbed,
            rows,
            args=args,
            sigma=sigma,
            population_shape=(sample_count, width),
        ),
        args.output,
    )

    manifest_path = write_manifest(
        args,
        population_shape=(sample_count, width),
        sigma=sigma,
        ids=ids,
        rows=rows,
    )
    print(f"\nSaved {len(rows)} perturbed adapter(s): {args.output}")
    print(f"Manifest: {manifest_path}")
    print("Next: render x_hat directly with experiments/generate_images.py")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

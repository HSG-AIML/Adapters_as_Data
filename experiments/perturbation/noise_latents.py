"""Perturb SANE latent payloads with Gaussian noise.

The noise model is

    z' = z + scale * sigma * N(0, 1)

where ``sigma`` is the global standard deviation of ``z`` over all samples and
tokens of the payload, so ``--noise-scale`` is a fraction of the latent space's
own spread. Original and perturbed statistics are reported for every input.

Each input keeps its filename inside ``--output-dir`` and stays a latent payload
for ``experiments/reconstruct_latents.py``; only ``z`` changes.

Example:

    uv run python experiments/perturbation/noise_latents.py \
        outputs/latents/latents_train.pt outputs/latents/latents_val.pt \
        --output-dir outputs/perturbation/latents_075 \
        --noise-scale 0.75

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


FORMAT_NAME = "sane.latents"
FORMAT_VERSION = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "latents",
        nargs="+",
        type=Path,
        help="One or more trusted latent payloads to perturb.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.75,
        help="Noise standard deviation as a fraction of the global z std "
        "(default: 0.75).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        help="Override the population spread instead of measuring it per payload.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for the noise draws (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing perturbed payloads.",
    )
    return parser.parse_args(argv)


def load_latent_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Latent payload not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Latent payload must contain a mapping: {path}")
    payload = dict(payload)
    if "z" not in payload:
        raise KeyError(f"Latent payload is missing the z field: {path}")
    latents = payload["z"]
    if not isinstance(latents, torch.Tensor) or latents.ndim != 3:
        raise ValueError(f"{path}: z must be a rank-3 tensor")
    return payload


def output_paths(args: argparse.Namespace) -> list[Path]:
    paths = [args.output_dir / path.name for path in args.latents]
    if len(set(paths)) != len(paths):
        raise ValueError(
            "Multiple inputs resolve to the same output filename; run them "
            "separately or use distinct input names"
        )
    for source, destination in zip(args.latents, paths):
        if source.resolve() == destination.resolve():
            raise ValueError(f"Output would overwrite the input payload: {source}")
    return paths


def build_result_payload(
    source: dict[str, Any],
    perturbed: torch.Tensor,
    *,
    noise_scale: float,
    noise_std: float,
    sigma: float,
    seed: int,
    source_path: Path,
) -> dict[str, Any]:
    payload = dict(source)  # shallow copy; tensors are shared except z
    payload.update(
        {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "z": perturbed,
            "space": "latent",
            "source_file": str(source_path.resolve()),
            "noise_scale": noise_scale,
            "noise_std": noise_std,
            "noise_sigma": sigma,
            "noise_seed": seed,
            "original_z_std": sigma,
        }
    )
    return payload


def write_manifest(
    args: argparse.Namespace,
    outputs: list[dict[str, Any]],
) -> Path:
    manifest = {
        "space": "latent",
        "noise_scale": args.noise_scale,
        "sigma_override": args.sigma,
        "seed": args.seed,
        "outputs": outputs,
    }
    path = args.output_dir / "noise_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> None:
    if args.noise_scale < 0:
        raise ValueError("--noise-scale must be non-negative")
    if args.sigma is not None and args.sigma < 0:
        raise ValueError("--sigma must be non-negative")

    destinations = output_paths(args)
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Perturbation output(s) already exist: {existing}. "
            "Pass --overwrite to replace them."
        )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for source_path, destination in zip(args.latents, destinations):
        payload = load_latent_payload(source_path)
        latents = payload["z"]
        split = payload.get("split", source_path.stem.removeprefix("latents_"))
        print(f"\n{source_path}")
        original_stats = summarize(latents)
        print_statistics(original_stats, f"{split} - original")

        sigma = original_stats["std"] if args.sigma is None else args.sigma
        perturbed, noise_std = add_gaussian_noise(
            latents,
            scale=args.noise_scale,
            sigma=sigma,
            generator=generator,
        )
        print(
            f"  noise: scale={args.noise_scale}, sigma={sigma:.6f}, "
            f"std={noise_std:.6f} ({args.noise_scale * 100:.1f}% of sigma)"
        )
        perturbed_stats = summarize(perturbed)
        print_statistics(perturbed_stats, f"{split} - perturbed")

        torch.save(
            build_result_payload(
                payload,
                perturbed,
                noise_scale=args.noise_scale,
                noise_std=noise_std,
                sigma=sigma,
                seed=args.seed,
                source_path=source_path,
            ),
            destination,
        )
        outputs.append(
            {
                "source": str(source_path.resolve()),
                "output": str(destination.resolve()),
                "split": split,
                "z_shape": list(original_stats["shape"]),
                "sigma": sigma,
                "noise_std": noise_std,
                "original_std": original_stats["std"],
                "perturbed_std": perturbed_stats["std"],
            }
        )
        print(f"  saved {destination}")

    manifest_path = write_manifest(args, outputs)
    print(f"\nSaved {len(outputs)} perturbed latent payload(s)")
    print(f"Manifest: {manifest_path}")
    print("Next: reconstruct the payloads with experiments/reconstruct_latents.py")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

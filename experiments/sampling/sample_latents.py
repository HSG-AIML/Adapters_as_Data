"""Sample new SANE latents from an encoded training population.

Two methods from the paper are available:

``naive``
    Fit an independent Gaussian to every coordinate of the flattened training
    latents and sample each coordinate independently.

``pca``
    Center and flatten the training latents, fit PCA with an economy SVD, draw
    independent Gaussian PCA coefficients using their empirical means and
    standard deviations, and project back to the token-shaped latent space.

Both methods apply ``--factor`` to the fitted standard deviations. A factor of
1 reproduces the observed marginal spread; larger values increase diversity.
The PCA implementation intentionally retains the original full economy-SVD
procedure.

Examples:

    uv run python experiments/sampling/sample_latents.py outputs/latents/latents_train.pt \
        --output outputs/sampling/latents_pca.pt \
        --method pca --components 256 --factor 1 --num-samples 9

    uv run python experiments/sampling/sample_latents.py outputs/latents/latents_train.pt \
        --output outputs/sampling/latents_naive.pt \
        --method naive --factor 1 --num-samples 9

The output is a standard latent payload for
``experiments/reconstruct_latents.py``. Its ``x``, ``p``, and ``mask`` rows are
sampled from the source population solely as reconstruction metadata; they are
not ground truth for the generated ``z``. Consequently, generated weights
should be rendered with ``--fields x_hat --no-scores``.

Torch payloads are pickle-backed and must come from trusted sources.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from sampling import (
    explained_variance_summary,
    fit_pca,
    sample_naive,
    sample_pca,
)


FORMAT_NAME = "sane.latents"
FORMAT_VERSION = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_latents", type=Path, help="Trusted training latent payload.")
    parser.add_argument("--output", type=Path, required=True, help="Sampled latent .pt file.")
    parser.add_argument("--method", choices=("naive", "pca"), default="pca")
    parser.add_argument("--num-samples", type=int, default=9)
    parser.add_argument(
        "--components",
        type=int,
        default=256,
        help="PCA components requested in pca mode (default: 256).",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=1.0,
        help="Multiplier on fitted coordinate/component standard deviations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Optionally save the fitted Gaussian/PCA parameters separately.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output and model files.",
    )
    return parser.parse_args(argv)


def load_source_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Training latent payload not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Latent payload must contain a mapping: {path}")
    payload = dict(payload)

    missing = [field for field in ("z", "x", "p") if field not in payload]
    if missing:
        raise KeyError(f"Training latent payload is missing fields: {missing}")
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
    if latents.shape[0] < 2:
        raise ValueError("Sampling requires at least two training latents")

    mask = payload.get("mask")
    if mask is not None and (
        not isinstance(mask, torch.Tensor) or mask.shape != latents.shape[:2]
    ):
        raise ValueError("mask must have shape [N, tokens]")
    ids = payload.get("ids")
    if ids is not None and len(ids) != latents.shape[0]:
        raise ValueError("ids must contain one entry per training latent")
    return payload



def _source_ids(payload: dict[str, Any], sample_count: int) -> list[Any]:
    values = payload.get("ids")
    if values is None:
        return list(range(sample_count))
    if isinstance(values, torch.Tensor):
        return values.cpu().tolist()
    return list(values)


def build_sample_payload(
    source: dict[str, Any],
    sampled_latents: torch.Tensor,
    reference_rows: torch.Tensor,
    *,
    method: str,
    factor: float,
    components: int | None,
    seed: int,
    source_path: Path,
) -> dict[str, Any]:
    sample_count = sampled_latents.shape[0]
    source_count = source["z"].shape[0]
    source_ids = _source_ids(source, source_count)
    reference_ids = [source_ids[row] for row in reference_rows.tolist()]

    mask = source.get("mask")
    sampled_mask = mask[reference_rows] if isinstance(mask, torch.Tensor) else None
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "z": sampled_latents,
        "x": source["x"][reference_rows],
        "p": source["p"][reference_rows],
        "mask": sampled_mask,
        "ids": [f"{method}_sample_{index:05d}" for index in range(sample_count)],
        "split": "sampled",
        "epoch": source.get("epoch"),
        "config": source.get("config"),
        "checkpoint": source.get("checkpoint"),
        "sampling_method": method,
        "n_samples": sample_count,
        "n_components": components,
        "pca_factor": factor if method == "pca" else None,
        "sampling_factor": factor,
        "sampling_seed": seed,
        "source_file": str(source_path.resolve()),
        "reference_rows": reference_rows,
        "reference_ids": reference_ids,
        "reference_note": (
            "x, p, and mask are sampled reconstruction metadata and do not "
            "correspond to the generated z"
        ),
    }



def write_manifest(
    args: argparse.Namespace,
    *,
    source_shape: tuple[int, ...],
    sample_shape: tuple[int, ...],
    effective_components: int | None,
    explained_variance: dict[str, float] | None,
) -> Path:
    manifest = {
        "source": str(args.train_latents.resolve()),
        "output": str(args.output.resolve()),
        "method": args.method,
        "num_samples": args.num_samples,
        "factor": args.factor,
        "requested_components": args.components if args.method == "pca" else None,
        "effective_components": effective_components,
        "seed": args.seed,
        "source_shape": list(source_shape),
        "sample_shape": list(sample_shape),
        "model_output": str(args.model_output.resolve()) if args.model_output else None,
        "cumulative_explained_variance": explained_variance,
    }
    path = args.output.with_name(f"{args.output.stem}_manifest.json")
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> None:
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if args.components < 1:
        raise ValueError("--components must be positive")
    if args.factor < 0:
        raise ValueError("--factor must be non-negative")
    if args.model_output == args.output:
        raise ValueError("--model-output must differ from --output")

    planned_outputs = [args.output]
    if args.model_output is not None:
        planned_outputs.append(args.model_output)
    existing = [path for path in planned_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Sampling output(s) already exist: {existing}. Pass --overwrite to replace them."
        )

    source = load_source_payload(args.train_latents)
    training_latents = source["z"]
    source_shape = tuple(training_latents.shape)
    sample_count, token_count, latent_dim = source_shape
    flattened = training_latents.float().reshape(sample_count, -1)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    print(
        f"Loaded {args.train_latents}: z={source_shape}, "
        f"flattened width={flattened.shape[1]:,}"
    )

    effective_components = None
    explained_variance = None
    if args.method == "naive":
        print("Fitting independent Gaussian parameters for every latent coordinate ...")
        sampled_flat, fitted_parameters = sample_naive(
            flattened,
            num_samples=args.num_samples,
            factor=args.factor,
            generator=generator,
        )
    else:
        effective_components = min(args.components, *flattened.shape)
        if effective_components != args.components:
            print(
                f"Clamped PCA components from {args.components} to "
                f"{effective_components}"
            )
        print(f"Fitting PCA with {effective_components} components using economy SVD ...")
        fitted = fit_pca(flattened, effective_components)
        sampled_flat, fitted_parameters = sample_pca(
            fitted,
            num_samples=args.num_samples,
            factor=args.factor,
            generator=generator,
        )
        explained_variance = explained_variance_summary(
            fitted["explained_variance_ratio"]
        )
        print("Cumulative explained variance:")
        for name, value in explained_variance.items():
            print(f"  {name.replace('_', '-')}: {100 * value:.2f}%")

    sampled_latents = sampled_flat.reshape(
        args.num_samples,
        token_count,
        latent_dim,
    ).to(training_latents.dtype)
    reference_rows = torch.randint(
        0,
        sample_count,
        (args.num_samples,),
        generator=generator,
    )
    sampled_payload = build_sample_payload(
        source,
        sampled_latents,
        reference_rows,
        method=args.method,
        factor=args.factor,
        components=effective_components,
        seed=args.seed,
        source_path=args.train_latents,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sampled_payload, args.output)
    if args.model_output is not None:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": args.method,
                "source_file": str(args.train_latents.resolve()),
                "source_shape": source_shape,
                "factor": args.factor,
                "components": effective_components,
                **fitted_parameters,
            },
            args.model_output,
        )

    manifest_path = write_manifest(
        args,
        source_shape=source_shape,
        sample_shape=tuple(sampled_latents.shape),
        effective_components=effective_components,
        explained_variance=explained_variance,
    )
    print(
        f"Sample statistics: mean={sampled_latents.float().mean():.6f}, "
        f"std={sampled_latents.float().std():.6f}"
    )
    print(f"Saved latent payload: {args.output}")
    if args.model_output is not None:
        print(f"Saved fitted sampler: {args.model_output}")
    print(f"Saved manifest: {manifest_path}")
    print("Next: reconstruct this payload with experiments/reconstruct_latents.py")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

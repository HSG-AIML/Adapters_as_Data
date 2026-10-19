"""Sample flat LoRA adapters directly in raw weight space.

This is the matched weight-space control for ``sample_latents.py``. It applies
the same independent-coordinate Gaussian or economy-SVD PCA operation to flat
adapter weights, but bypasses the SANE encoder and decoder entirely:

    flat training LoRAs -> naive/PCA sampling -> sampled x_hat -> image renderer

For the closest population match, use the ``x`` field from the reconstructed
training-latent payload. That guarantees the same rows used by latent sampling:

    uv run python experiments/sampling/sample_weights.py \
        outputs/reconstructions/x_and_x_hat_train.pt \
        --field x \
        --output outputs/sampling/weights_pca.pt \
        --method pca --components 256 --factor 1 --num-samples 9

The input may also be a raw tensor ``[N, W]`` instead of a dictionary. Outputs
already contain flat ``x`` and ``x_hat`` fields and therefore go directly to
``experiments/generate_images.py``; no latent reconstruction step is involved.
The reference rows in ``x`` are metadata only and are not targets for sampled
``x_hat``, so render with ``--fields x_hat --no-scores``.

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


FORMAT_NAME = "sane.reconstructed_weights"
FORMAT_VERSION = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "training_weights",
        type=Path,
        help="Trusted tensor or payload containing the flat training adapters.",
    )
    parser.add_argument(
        "--field",
        default="x",
        help="Weight field when the input is a dictionary (default: x).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("naive", "pca"), default="pca")
    parser.add_argument("--num-samples", type=int, default=9)
    parser.add_argument("--components", type=int, default=256)
    parser.add_argument(
        "--factor",
        type=float,
        default=1.0,
        help="Multiplier on fitted coordinate/component standard deviations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        help="Use only the first N input adapters when fitting the sampler.",
    )
    parser.add_argument(
        "--mmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Memory-map tensors from the Torch archive when possible (default: true).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Optionally save fitted Gaussian/PCA parameters separately.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output and model files.",
    )
    return parser.parse_args(argv)


def load_training_weights(
    path: Path,
    *,
    field: str,
    mmap: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Training weights not found: {path}")
    loaded = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=mmap,
    )
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
        raise ValueError("Training-weight input must be a tensor or dictionary")

    if not isinstance(weights, torch.Tensor) or weights.ndim < 2:
        raise ValueError(f"Weight field {field!r} must have a sample axis")
    flattened = weights.reshape(weights.shape[0], -1)
    if flattened.shape[0] < 2:
        raise ValueError("Weight sampling requires at least two training adapters")
    return flattened, metadata


def source_ids(metadata: dict[str, Any], sample_count: int) -> list[Any]:
    values = metadata.get("indices", metadata.get("ids"))
    if values is None:
        return list(range(sample_count))
    if len(values) < sample_count:
        raise ValueError(
            f"Input IDs contain {len(values)} entries for {sample_count} adapters"
        )
    if isinstance(values, torch.Tensor):
        return values[:sample_count].cpu().tolist()
    return list(values[:sample_count])


def sample_weights(
    flattened: torch.Tensor,
    *,
    method: str,
    num_samples: int,
    components: int,
    factor: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int | None, dict[str, float] | None]:
    fit_data = flattened.float()
    if method == "naive":
        samples, parameters = sample_naive(
            fit_data,
            num_samples=num_samples,
            factor=factor,
            generator=generator,
        )
        return samples, parameters, None, None

    effective_components = min(components, *fit_data.shape)
    if effective_components != components:
        print(f"Clamped PCA components from {components} to {effective_components}")
    fitted = fit_pca(fit_data, effective_components)
    samples, parameters = sample_pca(
        fitted,
        num_samples=num_samples,
        factor=factor,
        generator=generator,
    )
    explained = explained_variance_summary(fitted["explained_variance_ratio"])
    return samples, parameters, effective_components, explained


def write_manifest(
    args: argparse.Namespace,
    *,
    training_shape: tuple[int, ...],
    sample_shape: tuple[int, ...],
    effective_components: int | None,
    explained_variance: dict[str, float] | None,
) -> Path:
    manifest = {
        "source": str(args.training_weights.resolve()),
        "source_field": args.field,
        "output": str(args.output.resolve()),
        "method": args.method,
        "num_samples": args.num_samples,
        "factor": args.factor,
        "requested_components": args.components if args.method == "pca" else None,
        "effective_components": effective_components,
        "seed": args.seed,
        "limit": args.limit,
        "training_shape": list(training_shape),
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
    if args.limit is not None and args.limit < 2:
        raise ValueError("--limit must be at least 2")
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

    training_weights, metadata = load_training_weights(
        args.training_weights,
        field=args.field,
        mmap=args.mmap,
    )
    if args.limit is not None:
        training_weights = training_weights[: args.limit]
    training_count, weight_width = training_weights.shape
    ids = source_ids(metadata, training_count)
    print(
        f"Loaded {training_count} training adapters with {weight_width:,} weights each"
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    sampled, fitted_parameters, effective_components, explained_variance = sample_weights(
        training_weights,
        method=args.method,
        num_samples=args.num_samples,
        components=args.components,
        factor=args.factor,
        generator=generator,
    )
    if explained_variance is not None:
        print("Cumulative explained variance:")
        for name, value in explained_variance.items():
            print(f"  {name.replace('_', '-')}: {100 * value:.2f}%")

    reference_rows = torch.randint(
        0,
        training_count,
        (args.num_samples,),
        generator=generator,
    )
    reference_ids = [ids[row] for row in reference_rows.tolist()]
    generated_ids = [
        f"weight_{args.method}_sample_{index:05d}"
        for index in range(args.num_samples)
    ]
    output_payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "x": training_weights[reference_rows].clone(),
        "x_hat": sampled.to(training_weights.dtype),
        "indices": generated_ids,
        "ids": generated_ids,
        "split": "weight_sampled",
        "epoch": metadata.get("epoch"),
        "config": metadata.get("config"),
        "space": "weight",
        "sampling_method": args.method,
        "n_samples": args.num_samples,
        "n_components": effective_components,
        "pca_factor": args.factor if args.method == "pca" else None,
        "sampling_factor": args.factor,
        "sampling_seed": args.seed,
        "source_file": str(args.training_weights.resolve()),
        "source_field": args.field,
        "reference_rows": reference_rows,
        "reference_ids": reference_ids,
        "reference_note": (
            "x contains sampled reference adapters and is not ground truth for x_hat"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)
    if args.model_output is not None:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": args.method,
                "source_file": str(args.training_weights.resolve()),
                "source_field": args.field,
                "training_shape": tuple(training_weights.shape),
                "factor": args.factor,
                "components": effective_components,
                **fitted_parameters,
            },
            args.model_output,
        )

    manifest_path = write_manifest(
        args,
        training_shape=tuple(training_weights.shape),
        sample_shape=tuple(sampled.shape),
        effective_components=effective_components,
        explained_variance=explained_variance,
    )
    print(
        f"Sample statistics: mean={sampled.float().mean():.6f}, "
        f"std={sampled.float().std():.6f}"
    )
    print(f"Saved sampled weight payload: {args.output}")
    if args.model_output is not None:
        print(f"Saved fitted sampler: {args.model_output}")
    print(f"Saved manifest: {manifest_path}")
    print("Next: render x_hat directly with experiments/generate_images.py")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

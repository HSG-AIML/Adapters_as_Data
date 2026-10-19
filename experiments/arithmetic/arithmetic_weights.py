"""Apply pairwise arithmetic directly to two flat LoRA weight vectors.

This is the matched raw-weight control for ``arithmetic_latents.py``. It uses the
same shared operations and defaults, but bypasses SANE reconstruction:

    flat LoRA A/B -> arithmetic -> x_hat -> image renderer

For a matched comparison, use the ``x`` field from the reconstruction of the
same latent population and select the same ``--sample-a`` and ``--sample-b``
rows used by the latent script.

Example:

    uv run python experiments/arithmetic/arithmetic_weights.py \
        outputs/reconstructions/x_and_x_hat_val.pt \
        --field x \
        --output-dir outputs/arithmetic/weights \
        --sample-a 6 --sample-b 8

Each result is a one-sample flat-weight payload for ``experiments/generate_images.py``.
Its ``x`` field is sample A as a reference, not ground truth for the arithmetic
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

from arithmetic import (
    DEFAULT_ADD_SCALES,
    DEFAULT_OPERATIONS,
    OPERATION_NAMES,
    apply_operation,
    operation_filename,
    operation_specs,
    sample_ids,
    tensor_stats,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-a", type=int, default=6)
    parser.add_argument("--sample-b", type=int, default=8)
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=OPERATION_NAMES,
        default=DEFAULT_OPERATIONS,
        help="Operations to generate (default: all).",
    )
    parser.add_argument(
        "--add-scales",
        nargs="*",
        type=float,
        default=DEFAULT_ADD_SCALES,
        help="Scales expanded when add is selected (default: 0.5 1.5).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing arithmetic payloads.",
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
    result: torch.Tensor,
    *,
    operation: str,
    label: str,
    kwargs: dict[str, float],
    sample_a: int,
    sample_b: int,
    id_a: Any,
    id_b: Any,
    source_path: Path,
    source_field: str,
) -> dict[str, Any]:
    result_id = operation_filename(operation, kwargs, sample_a, sample_b).removesuffix(".pt")
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "x": sample_a_weights.unsqueeze(0),
        "x_hat": result.to(sample_a_weights.dtype).unsqueeze(0),
        "indices": [result_id],
        "ids": [result_id],
        "split": f"weight_arithmetic_{operation}",
        "epoch": source.get("epoch"),
        "config": source.get("config"),
        "checkpoint": source.get("checkpoint"),
        "source_file": str(source_path.resolve()),
        "source_field": source_field,
        "space": "weight",
        "arith_operation": operation,
        "arith_label": label,
        "arith_kwargs": kwargs,
        "arith_sample_a": sample_a,
        "arith_sample_b": sample_b,
        "arith_id_a": id_a,
        "arith_id_b": id_b,
        "reference_note": (
            "x contains sample A as a reference and is not ground truth for x_hat"
        ),
    }


def write_manifest(
    args: argparse.Namespace,
    *,
    source_shape: tuple[int, ...],
    id_a: Any,
    id_b: Any,
    outputs: list[dict[str, Any]],
) -> Path:
    manifest = {
        "space": "weight",
        "source": str(args.weights.resolve()),
        "source_field": args.field,
        "source_shape": list(source_shape),
        "sample_a": args.sample_a,
        "sample_b": args.sample_b,
        "id_a": id_a,
        "id_b": id_b,
        "operations": outputs,
    }
    path = args.output_dir / "arithmetic_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> None:
    weights, metadata = load_weight_population(args.weights, args.field)
    validate_sample_pair(args.sample_a, args.sample_b, weights.shape[0])
    specs = operation_specs(args.operations, args.add_scales)
    ids = sample_ids(metadata, weights.shape[0])
    id_a = ids[args.sample_a]
    id_b = ids[args.sample_b]

    sample_a = weights[args.sample_a].float()
    sample_b = weights[args.sample_b].float()
    l2_distance = (sample_a - sample_b).norm().item()
    cosine = torch.nn.functional.cosine_similarity(
        sample_a.reshape(1, -1),
        sample_b.reshape(1, -1),
    ).item()
    print(f"Loaded {args.weights}: weights={tuple(weights.shape)}")
    print(f"A row={args.sample_a}, id={id_a}: {tensor_stats(sample_a)}")
    print(f"B row={args.sample_b}, id={id_b}: {tensor_stats(sample_b)}")
    print(f"A to B: L2={l2_distance:.4f}, cosine={cosine:.6f}")

    planned = [
        args.output_dir / operation_filename(operation, kwargs, args.sample_a, args.sample_b)
        for operation, kwargs in specs
    ]
    existing = [path for path in planned if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Arithmetic output(s) already exist: {existing}. Pass --overwrite to replace them."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for (operation, kwargs), output_path in zip(specs, planned):
        result, label = apply_operation(sample_a, sample_b, operation, **kwargs)
        result_payload = build_result_payload(
            metadata,
            weights[args.sample_a],
            result,
            operation=operation,
            label=label,
            kwargs=kwargs,
            sample_a=args.sample_a,
            sample_b=args.sample_b,
            id_a=id_a,
            id_b=id_b,
            source_path=args.weights,
            source_field=args.field,
        )
        torch.save(result_payload, output_path)
        outputs.append(
            {
                "operation": operation,
                "label": label,
                "kwargs": kwargs,
                "output": str(output_path.resolve()),
            }
        )
        print(f"{label:20s} {tensor_stats(result)} -> {output_path}")

    manifest_path = write_manifest(
        args,
        source_shape=tuple(weights.shape),
        id_a=id_a,
        id_b=id_b,
        outputs=outputs,
    )
    print(f"Saved {len(outputs)} weight arithmetic payload(s)")
    print(f"Manifest: {manifest_path}")
    print("Next: render x_hat directly with experiments/generate_images.py")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

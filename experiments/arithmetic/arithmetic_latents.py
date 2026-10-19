"""Apply pairwise arithmetic to two token-shaped SANE latents.

The available operations are shared with ``arithmetic_weights.py``:

- ``diff_a_minus_b``: ``A - B``
- ``diff_b_minus_a``: ``B - A``
- ``sum``: ``A + B``
- ``mean``: ``(A + B) / 2``
- ``add``: ``A + scale * B`` for each requested scale

Each result is written as a one-sample latent payload for
``experiments/reconstruct_latents.py``. Sample A supplies ``x``, ``p``, and
``mask`` reconstruction metadata, but its ``x`` is not ground truth for the
arithmetic result.

Example:

    uv run python experiments/arithmetic/arithmetic_latents.py \
        outputs/latents/latents_val.pt \
        --output-dir outputs/arithmetic/latents \
        --sample-a 6 --sample-b 8

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


FORMAT_NAME = "sane.latents"
FORMAT_VERSION = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("latents", type=Path, help="Trusted latent population payload.")
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
) -> dict[str, Any]:
    mask = source.get("mask")
    result_id = operation_filename(operation, kwargs, sample_a, sample_b).removesuffix(".pt")
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "z": result.to(source["z"].dtype).unsqueeze(0),
        "x": source["x"][sample_a].unsqueeze(0),
        "p": source["p"][sample_a].unsqueeze(0),
        "mask": mask[sample_a].unsqueeze(0) if mask is not None else None,
        "ids": [result_id],
        "split": f"arithmetic_{operation}",
        "epoch": source.get("epoch"),
        "config": source.get("config"),
        "checkpoint": source.get("checkpoint"),
        "source_file": str(source_path.resolve()),
        "space": "latent",
        "arith_operation": operation,
        "arith_label": label,
        "arith_kwargs": kwargs,
        "arith_sample_a": sample_a,
        "arith_sample_b": sample_b,
        "arith_id_a": id_a,
        "arith_id_b": id_b,
        "reference_note": (
            "x, p, and mask come from sample A and are reconstruction metadata, "
            "not ground truth for the arithmetic latent"
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
        "space": "latent",
        "source": str(args.latents.resolve()),
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
    payload = load_latent_population(args.latents)
    latents = payload["z"]
    validate_sample_pair(args.sample_a, args.sample_b, latents.shape[0])
    specs = operation_specs(args.operations, args.add_scales)
    ids = sample_ids(payload, latents.shape[0])
    id_a = ids[args.sample_a]
    id_b = ids[args.sample_b]

    sample_a = latents[args.sample_a].float()
    sample_b = latents[args.sample_b].float()
    l2_distance = (sample_a - sample_b).norm().item()
    cosine = torch.nn.functional.cosine_similarity(
        sample_a.reshape(1, -1),
        sample_b.reshape(1, -1),
    ).item()
    print(f"Loaded {args.latents}: z={tuple(latents.shape)}")
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
            payload,
            result,
            operation=operation,
            label=label,
            kwargs=kwargs,
            sample_a=args.sample_a,
            sample_b=args.sample_b,
            id_a=id_a,
            id_b=id_b,
            source_path=args.latents,
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
        source_shape=tuple(latents.shape),
        id_a=id_a,
        id_b=id_b,
        outputs=outputs,
    )
    print(f"Saved {len(outputs)} latent arithmetic payload(s)")
    print(f"Manifest: {manifest_path}")
    print("Next: reconstruct the payloads with experiments/reconstruct_latents.py")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

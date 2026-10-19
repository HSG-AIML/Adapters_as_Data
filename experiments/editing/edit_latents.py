"""Edit SANE latents along fitted attribute directions.

For every requested attribute a direction is fitted in the flattened latent
space by ridge regression against the binary attribute labels, optionally
debiased against nuisance attributes, and applied as a strength sweep:

    z' = z + alpha * ||z|| * d

Each sweep is written as a multi-row latent payload for
``experiments/reconstruct_latents.py``. Row order within a file is the order of
``--strengths``, and the ``alpha = 0`` row is the unedited control. Sample A
supplies ``x``, ``p``, and ``mask`` reconstruction metadata; that ``x`` is not
ground truth for the edited latents, so render the reconstructions with
``--fields x_hat --no-scores``.

Strengths are relative by default: ``alpha`` is a fraction of ``||z||``, so
``--strengths 0.5`` moves the latent by half its own norm regardless of the
latent's absolute scale, and the numbers stay comparable across samples and
models. Directions are oriented so that positive ``alpha`` exaggerates the
attribute.

``--attributes`` and ``--control-attributes`` are fitted and rendered
identically; the split only records which attributes the direction screen
predicted would work. Rendering both groups is what turns the screen from an
unvalidated proxy into a result: if the control group shows no visible change at
the strength where the target group clearly works, the screen predicts edit
quality.

Example:

    uv run python experiments/editing/edit_latents.py \
        outputs/latents/latents_train.pt \
        --identity-df path/to/identity_df.pt \
        --output-dir outputs/editing/latents \
        --attributes Bald --control-attributes Oval_Face \
        --debias Bald=No_Beard,Male \
        --rows 1

Torch payloads are pickle-backed and must come from trusted sources.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from sane.postprocessing import resolve_device

from editing import (
    DEFAULT_AUC_WARNING,
    DEFAULT_GRAM_CHUNK,
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_HOLDOUT_SEED,
    DEFAULT_HOLDOUT_SPLITS,
    DEFAULT_RIDGE_LAMBDA,
    DirectionFit,
    attribute_group,
    fit_directions,
    load_identity_frame,
    parse_debias_map,
    print_direction_table,
    resolve_labels,
    tile_rows,
    unique_attributes,
)


FORMAT_NAME = "sane.latents"
FORMAT_VERSION = 1
DEFAULT_STRENGTHS = (0.0, 0.4, 0.45, 0.5, 0.55, 0.6)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("latents", type=Path, help="Trusted latent population payload.")
    parser.add_argument(
        "--identity-df",
        type=Path,
        required=True,
        help="Identity dataframe whose row order matches the latent payload.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--attributes",
        nargs="+",
        required=True,
        help="Attributes to edit.",
    )
    parser.add_argument(
        "--control-attributes",
        nargs="*",
        default=(),
        help="Additional attributes the direction screen predicts will fail.",
    )
    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        default=(1,),
        help="Latent rows to edit (default: 1).",
    )
    parser.add_argument(
        "--debias",
        action="append",
        metavar="TARGET=NUISANCE[,NUISANCE...]",
        help="Remove a nuisance attribute's component from a target direction.",
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=DEFAULT_STRENGTHS,
        help="Edit strengths applied in sweep order (default: 0 0.4 0.45 0.5 0.55 0.6).",
    )
    parser.add_argument(
        "--strength-mode",
        choices=("relative", "absolute"),
        default="relative",
        help="relative: alpha is a fraction of ||z||; absolute: alpha scales the "
        "unit direction directly (default: relative).",
    )
    parser.add_argument(
        "--ridge-lambda",
        type=float,
        default=DEFAULT_RIDGE_LAMBDA,
        help=f"Ridge regularization strength (default: {DEFAULT_RIDGE_LAMBDA}).",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
        help=f"Fraction held out of each diagnostic fit (default: {DEFAULT_HOLDOUT_FRACTION}).",
    )
    parser.add_argument(
        "--holdout-splits",
        type=int,
        default=DEFAULT_HOLDOUT_SPLITS,
        help=f"Repeated random splits for the held-out AUC (default: {DEFAULT_HOLDOUT_SPLITS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_HOLDOUT_SEED,
        help=f"Seed for the diagnostic splits (default: {DEFAULT_HOLDOUT_SEED}).",
    )
    parser.add_argument(
        "--auc-warning",
        type=float,
        default=DEFAULT_AUC_WARNING,
        help=f"Flag directions below this held-out AUC (default: {DEFAULT_AUC_WARNING}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Fit the directions on only the first N latents.",
    )
    parser.add_argument(
        "--gram-chunk",
        type=int,
        default=DEFAULT_GRAM_CHUNK,
        help=f"Rows per chunk when building K (default: {DEFAULT_GRAM_CHUNK}).",
    )
    parser.add_argument(
        "--save-directions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store each unit direction in its payload (default: true).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. cuda:0 or cpu (default: auto).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing edit payloads.",
    )
    return parser.parse_args(argv)


def load_latent_population(path: Path, limit: int | None) -> dict[str, Any]:
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

    if limit is not None:
        if limit < 2:
            raise ValueError("--limit must be at least 2")
        if limit > latents.shape[0]:
            raise IndexError(
                f"--limit {limit} exceeds the {latents.shape[0]} available latents"
            )
        for field in ("z", "x", "p", "mask"):
            value = payload.get(field)
            if isinstance(value, torch.Tensor):
                payload[field] = value[:limit]
        ids = payload.get("ids")
        if ids is not None:
            payload["ids"] = ids[:limit]
    if payload["z"].shape[0] < 2:
        raise ValueError("Direction fitting requires at least two latents")
    return payload


def edit_filename(attribute: str, sample: int) -> str:
    return f"edit_{attribute}_sample{sample}.pt"


def build_edit_payload(
    source: dict[str, Any],
    edited: torch.Tensor,
    direction: torch.Tensor | None,
    *,
    attribute: str,
    sample: int,
    strengths: Sequence[float],
    strength_mode: str,
    latent_norm: float,
    args: argparse.Namespace,
    debias_map: Mapping[str, Sequence[str]],
    diagnostics: dict[str, Any],
    fitted_samples: int,
) -> dict[str, Any]:
    step_count = edited.shape[0]
    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "z": edited.to(source["z"].dtype),
        "ids": [
            f"edit_{attribute}_s{sample}_a{strength:+.3f}" for strength in strengths
        ],
        "split": f"edited_{attribute}",
        "epoch": source.get("epoch"),
        "config": source.get("config"),
        "checkpoint": source.get("checkpoint"),
        "source_file": str(args.latents.resolve()),
        "space": "latent",
        "edit_attribute": attribute,
        "edit_sample_idx": sample,
        "edit_strengths": list(strengths),
        "edit_strength_mode": strength_mode,
        "edit_z_norm": latent_norm,
        "edit_ridge_lambda": args.ridge_lambda,
        "edit_debias": list(debias_map.get(attribute, ())),
        "edit_n_fitted": fitted_samples,
        "edit_sign_convention": "positive alpha exaggerates the attribute",
        "edit_diagnostics": diagnostics,
        "reference_note": (
            "x, p, and mask are tiled from the edited sample and are "
            "reconstruction metadata, not ground truth for the edited z"
        ),
    }
    for field in ("x", "p", "mask"):
        value = source.get(field)
        payload[field] = (
            tile_rows(value[sample], step_count).to(value.dtype)
            if isinstance(value, torch.Tensor)
            else None
        )
    if direction is not None:
        payload["edit_direction"] = direction.reshape(*source["z"].shape[1:]).cpu()
    return payload


def write_manifest(
    args: argparse.Namespace,
    *,
    source_shape: tuple[int, ...],
    fitted_samples: int,
    strengths: Sequence[float],
    debias_map: Mapping[str, Sequence[str]],
    fit: DirectionFit,
    outputs: list[dict[str, Any]],
) -> Path:
    manifest = {
        "space": "latent",
        "source": str(args.latents.resolve()),
        "source_shape": list(source_shape),
        "identity_df": str(args.identity_df.resolve()),
        "n_samples_fitted": fitted_samples,
        "flat_dim": int(source_shape[1] * source_shape[2]),
        "attributes": list(args.attributes),
        "control_attributes": list(args.control_attributes),
        "rows": list(args.rows),
        "ridge_lambda": args.ridge_lambda,
        "strength_mode": args.strength_mode,
        "strengths": list(strengths),
        "holdout_fraction": args.holdout_fraction,
        "holdout_splits": args.holdout_splits,
        "seed": args.seed,
        "orient_positive_adds": True,
        "debias_map": {key: list(value) for key, value in debias_map.items()},
        "diagnostics": fit.diagnostics,
        "outputs": outputs,
    }
    path = args.output_dir / "edit_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> None:
    if args.ridge_lambda < 0:
        raise ValueError("--ridge-lambda must be non-negative")
    if args.holdout_splits < 1:
        raise ValueError("--holdout-splits must be positive")
    if not 0 < args.holdout_fraction < 1:
        raise ValueError("--holdout-fraction must lie strictly between 0 and 1")
    if args.gram_chunk < 1:
        raise ValueError("--gram-chunk must be positive")
    strengths = list(dict.fromkeys(args.strengths))
    if not strengths:
        raise ValueError("At least one --strengths value is required")
    rows = list(dict.fromkeys(args.rows))
    targets = unique_attributes(args.attributes, args.control_attributes)
    debias_map = parse_debias_map(args.debias)

    device = resolve_device(args.device)
    payload = load_latent_population(args.latents, args.limit)
    latents = payload["z"].float()
    sample_count, token_count, latent_dim = latents.shape
    width = token_count * latent_dim
    print(f"Loaded {args.latents}: z={tuple(latents.shape)} on {device}")
    print(f"  flattened width D={width:,}")

    invalid = [row for row in rows if row < 0 or row >= sample_count]
    if invalid:
        raise IndexError(f"--rows {invalid} are outside 0..{sample_count - 1}")

    planned = {
        (attribute, sample): args.output_dir / edit_filename(attribute, sample)
        for sample in rows
        for attribute in targets
    }
    existing = [path for path in planned.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Edit output(s) already exist: {existing}. Pass --overwrite to replace them."
        )

    frame = load_identity_frame(args.identity_df)
    print(f"Loaded {args.identity_df}: {len(frame)} rows")
    fitted_attributes, labels = resolve_labels(frame, targets, debias_map, sample_count)
    print(
        f"Fitting {len(fitted_attributes)} direction(s) for "
        f"{len(targets)} target attribute(s)"
    )

    flattened = latents.reshape(sample_count, width)
    edit_rows = {row: flattened[row].clone() for row in rows}
    # Centre in place: the rows to edit are already copied out, and the payload's
    # z is only read for dtype and token shape from here on.
    flattened -= flattened.mean(dim=0, keepdim=True)

    fit = fit_directions(
        flattened,
        labels,
        targets,
        fitted_attributes,
        debias_map=debias_map,
        ridge_lambda=args.ridge_lambda,
        holdout_fraction=args.holdout_fraction,
        splits=args.holdout_splits,
        seed=args.seed,
        device=device,
        chunk=args.gram_chunk,
    )
    print_direction_table(fit, auc_warning=args.auc_warning)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for sample in rows:
        latent = edit_rows[sample].to(device)
        latent_norm = float(latent.norm().item())
        print(f"\nSample {sample} (||z|| = {latent_norm:.2f})")

        for attribute in targets:
            direction = fit.directions[attribute]
            unit = latent_norm if args.strength_mode == "relative" else 1.0
            edited = torch.stack(
                [
                    (latent + strength * unit * direction)
                    .reshape(token_count, latent_dim)
                    .cpu()
                    for strength in strengths
                ],
                dim=0,
            )
            output_path = planned[(attribute, sample)]
            torch.save(
                build_edit_payload(
                    payload,
                    edited,
                    fit.directions[attribute] if args.save_directions else None,
                    attribute=attribute,
                    sample=sample,
                    strengths=strengths,
                    strength_mode=args.strength_mode,
                    latent_norm=latent_norm,
                    args=args,
                    debias_map=debias_map,
                    diagnostics=fit.diagnostics[attribute],
                    fitted_samples=sample_count,
                ),
                output_path,
            )
            outputs.append(
                {
                    "attribute": attribute,
                    "group": attribute_group(
                        attribute,
                        args.attributes,
                        args.control_attributes,
                    ),
                    "sample_index": sample,
                    "n_rows": len(strengths),
                    "auc_holdout": fit.diagnostics[attribute]["auc_holdout_mean"],
                    "output": str(output_path.resolve()),
                }
            )
            print(
                f"  {attribute:<20} AUC "
                f"{fit.diagnostics[attribute]['auc_holdout_mean']:.3f} -> "
                f"{output_path.name} ({len(strengths)} rows)"
            )

    manifest_path = write_manifest(
        args,
        source_shape=tuple(latents.shape),
        fitted_samples=sample_count,
        strengths=strengths,
        debias_map=debias_map,
        fit=fit,
        outputs=outputs,
    )
    print(f"\nSaved {len(outputs)} latent edit sweep(s)")
    print(f"Manifest: {manifest_path}")
    print(f"Row order within every file = strengths {strengths}")
    if 0.0 in strengths:
        print(
            f"Row {strengths.index(0.0)} (alpha = 0) is the unedited reconstruction, "
            "the control."
        )
    print("Next: reconstruct the payloads with experiments/reconstruct_latents.py")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

"""Edit flat LoRA adapters along fitted attribute directions.

This is the matched raw-weight control for ``edit_latents.py``. It uses the same
ridge fit, the same debiasing, the same held-out diagnostics, and the same sweep,
but bypasses SANE entirely:

    flat LoRA -> attribute direction -> theta' -> image renderer

    theta' = theta + alpha * unit * d

Three things differ from the latent version, and each matters for the comparison
to be a control rather than a different experiment:

1. Row alignment is checked, not assumed. The weight population, the identity
   dataframe, and the latents share a row order, so adapter ``i`` is the same
   identity in each; a silent off-by-one would turn the control into a comparison
   of two different people. Passing ``--align-with-latents`` fixes the population
   size to the one ``edit_latents.py`` fitted on and compares the stored token
   sequences against the raw weight rows before anything is fitted. The dense
   tokenizer chunks the flattened adapter in order and pads only the final token,
   so the first ``D`` entries of a stored token sequence must reproduce the raw
   weight row.

2. Strengths default to the class-mean separation along the fitted direction:
   ``--strength-mode separation`` means ``alpha = 1`` moves the sample from the
   negative-class mean to the positive-class mean. Weight space and latent space
   have no common scale, so a displacement of "half the norm" means different
   things in each, whereas the separation unit is defined by the population's own
   spread along the attribute axis. Use ``relative_norm`` to reproduce the latent
   script's convention exactly.

3. Every edit is reported in units of the global weight standard deviation, the
   unit in which raw weight space was found to tolerate perturbations only up to
   roughly ``0.5`` and to collapse by ``0.75-1.0``. If a displacement is already
   outside that regime, a failed edit says nothing about attribute structure; it
   says the adapter was destroyed.

There is no decode step in weight space, so ``reconstruct_latents.py`` is
skipped: each sweep is a flat-weight payload for ``experiments/generate_images.py``.
Its ``x`` field is the unedited adapter tiled over the sweep, not ground truth
for ``x_hat``, so render with ``--fields x_hat --no-scores``.

Example:

    uv run python experiments/editing/edit_weights.py \
        outputs/reconstructions/x_and_x_hat_train.pt \
        --field x \
        --identity-df path/to/identity_df.pt \
        --output-dir outputs/editing/weights \
        --attributes Bald --control-attributes Oval_Face \
        --rows 1

Torch payloads are pickle-backed and must come from trusted sources.
"""

from __future__ import annotations

import argparse
import json
import math
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


FORMAT_NAME = "sane.reconstructed_weights"
FORMAT_VERSION = 1
DEFAULT_STRENGTHS = (0.0, 4.0, 5.0, 6.0, 8.0, 10.0)
STRENGTH_MODES = ("separation", "sigma_proj", "relative_norm", "absolute")
DEFAULT_SAFE_NOISE_SCALE = 0.5
DEFAULT_COLLAPSE_NOISE_SCALE = 0.75


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
    parser.add_argument(
        "--identity-df",
        type=Path,
        required=True,
        help="Identity dataframe whose row order matches the weight population.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--attributes",
        nargs="+",
        required=True,
        help="Attributes to edit; mirror the latent run for a matched control.",
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
        help="Adapter rows to edit; use the latent run's rows (default: 1).",
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
        help="Edit strengths applied in sweep order (default: 0 4 5 6 8 10).",
    )
    parser.add_argument(
        "--strength-mode",
        choices=STRENGTH_MODES,
        default="separation",
        help="separation: alpha class separations along d; sigma_proj: alpha "
        "projection standard deviations; relative_norm: alpha * ||theta||; "
        "absolute: alpha on the unit direction (default: separation).",
    )
    parser.add_argument(
        "--align-with-latents",
        type=Path,
        help="Latent payload fixing the fitted population size and verifying row "
        "alignment against the stored token sequences.",
    )
    parser.add_argument(
        "--verify-alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare stored token sequences against the raw weight rows "
        "(default: true, requires --align-with-latents).",
    )
    parser.add_argument(
        "--verify-row-count",
        type=int,
        default=6,
        help="Leading rows probed during alignment verification (default: 6).",
    )
    parser.add_argument(
        "--verify-tolerance",
        type=float,
        default=1e-4,
        help="Maximum absolute difference accepted as an exact match (default: 1e-4).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Fit on only the first N adapters; ignored with --align-with-latents.",
    )
    parser.add_argument(
        "--expected-weight-dim",
        type=int,
        help="Fail early unless each adapter has this many weights "
        "(rank-1: 99648, rank-2: 199296).",
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
        "--safe-noise-scale",
        type=float,
        default=DEFAULT_SAFE_NOISE_SCALE,
        help="Perturbation scale above which detail loss was observed "
        f"(default: {DEFAULT_SAFE_NOISE_SCALE}).",
    )
    parser.add_argument(
        "--collapse-noise-scale",
        type=float,
        default=DEFAULT_COLLAPSE_NOISE_SCALE,
        help="Perturbation scale at which weight space was found to collapse "
        f"(default: {DEFAULT_COLLAPSE_NOISE_SCALE}).",
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
        "--mmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Memory-map tensors from the Torch archive when possible (default: true).",
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


def _load_torch_file(path: Path, mmap: bool) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    except (TypeError, RuntimeError, ValueError) as error:
        if not mmap:
            raise
        print(f"  mmap unavailable ({type(error).__name__}); loading fully")
        return torch.load(path, map_location="cpu", weights_only=False)


def load_weight_population(
    path: Path,
    *,
    field: str,
    limit: int | None,
    mmap: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load the adapter matrix, materializing only the rows actually needed."""

    if not path.is_file():
        raise FileNotFoundError(f"Weight payload not found: {path}")
    loaded = _load_torch_file(path, mmap)
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
    weights = weights.reshape(weights.shape[0], -1)
    print(f"  on disk: {tuple(weights.shape)}")

    if limit is not None:
        if limit < 2:
            raise ValueError("Fitting requires at least two adapters")
        if limit > weights.shape[0]:
            raise IndexError(
                f"Asked for {limit} rows, the population has {weights.shape[0]}"
            )
        weights = weights[:limit]
    if weights.shape[0] < 2:
        raise ValueError("Direction fitting requires at least two adapters")
    return weights.float().clone(), metadata


def read_alignment_probes(
    path: Path,
    probe_rows: Sequence[int],
    *,
    mmap: bool,
) -> tuple[int, dict[int, torch.Tensor]]:
    """Read the fitted population size and a few stored token sequences.

    Returns the latent sample count and, per probed row, the flattened token
    sequence to compare the raw weight row against.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Latent payload not found: {path}")
    payload = _load_torch_file(path, mmap)
    if not isinstance(payload, Mapping) or "z" not in payload:
        raise ValueError(f"Latent payload must contain a mapping with z: {path}")

    sample_count = payload["z"].shape[0]
    tokens = payload.get("x")
    if tokens is None:
        print("  latents carry no x (tokenized weights); alignment cannot be verified")
        return sample_count, {}
    probes = {
        row: tokens[row].reshape(-1).float().clone()
        for row in probe_rows
        if 0 <= row < sample_count
    }
    return sample_count, probes


def verify_alignment(
    weights: torch.Tensor,
    probes: Mapping[int, torch.Tensor],
    tolerance: float,
) -> str:
    """Check that latent row ``i`` and weight row ``i`` are the same adapter.

    A near-perfect correlation with a large absolute difference means the rows
    are aligned but the stored tokens were normalised; that is reported and
    accepted.
    """

    if not probes:
        print("  [skip] no probes available")
        return "first N rows"

    width = weights.shape[1]
    print(f"  {'row':>6}{'max|diff|':>13}{'corr':>10}   verdict")
    failures = []
    for row, tokens in sorted(probes.items()):
        if tokens.numel() < width:
            print(
                f"  [skip] row {row}: token sequence has {tokens.numel()} entries "
                f"< D={width}"
            )
            return "first N rows"
        stored, raw = tokens[:width], weights[row]
        difference = float((stored - raw).abs().max())
        centered_stored = stored - stored.mean()
        centered_raw = raw - raw.mean()
        correlation = float(
            (centered_stored @ centered_raw)
            / (centered_stored.norm() * centered_raw.norm() + 1e-30)
        )
        if difference <= tolerance:
            verdict = "exact match"
        elif correlation > 0.999:
            verdict = "aligned, tokens rescaled"
        else:
            verdict = "MISMATCH"
            failures.append(row)
        print(f"  {row:6d}{difference:13.3e}{correlation:10.5f}   {verdict}")

    if failures:
        raise RuntimeError(
            f"Row alignment check failed on rows {failures}: the latent at that "
            "index does not correspond to the same adapter as the weight "
            "population at that index. Fix the indexing before going further; "
            "every number downstream would otherwise describe a different "
            "identity than the latent experiment did. Pass --no-verify-alignment "
            "only if you are certain the stored tokens are not comparable to the "
            "raw weights."
        )
    return "first N rows, verified against stored tokens"


def strength_unit(
    mode: str,
    diagnostics: Mapping[str, Any],
    weight_norm: float,
) -> float:
    units = {
        "separation": diagnostics["separation"],
        "sigma_proj": diagnostics["spread"],
        "relative_norm": weight_norm,
        "absolute": 1.0,
    }
    return float(units[mode])


def applied_strengths(
    strengths: Sequence[float],
    *,
    unit: float,
    weight_norm: float,
    weight_std: float,
    width: int,
) -> list[dict[str, float]]:
    """Describe each displacement, including its equivalent noise scale.

    A perturbation ``a * sigma * eps`` with ``eps ~ N(0, I)`` has norm
    ``a * sigma * sqrt(D)``, which is the unit in which weight-space robustness
    was measured.
    """

    root_width = math.sqrt(width)
    applied = []
    for strength in strengths:
        displacement = abs(strength * unit)
        applied.append(
            {
                "alpha": float(strength),
                "displacement_norm": displacement,
                "frac_of_theta_norm": displacement / (weight_norm + 1e-12),
                "noise_equivalent_a": displacement / (weight_std * root_width + 1e-12),
            }
        )
    return applied


def report_applied_strengths(
    applied: Sequence[Mapping[str, float]],
    *,
    safe_scale: float,
    collapse_scale: float,
) -> float:
    print(f"    {'alpha':>7}{'||dtheta||':>13}{'% of ||th||':>13}{'noise a':>10}")
    worst = 0.0
    for entry in applied:
        worst = max(worst, entry["noise_equivalent_a"])
        print(
            f"    {entry['alpha']:7.2f}{entry['displacement_norm']:13.4f}"
            f"{entry['frac_of_theta_norm'] * 100:12.1f}%"
            f"{entry['noise_equivalent_a']:10.3f}"
        )
    if worst > collapse_scale:
        print(
            f"    [WARNING] largest edit is equivalent to noise scale a = {worst:.2f}, "
            f"past the a = {collapse_scale} at which weight space collapses. A "
            "failure at this strength is the adapter being destroyed, not the "
            "direction being wrong."
        )
    elif worst > safe_scale:
        print(
            f"    [note] largest edit reaches a = {worst:.2f}, into the range where "
            "detail loss was observed."
        )
    return worst


def edit_filename(attribute: str, sample: int) -> str:
    return f"wedit_{attribute}_sample{sample}.pt"


def build_edit_payload(
    metadata: Mapping[str, Any],
    original: torch.Tensor,
    edited: torch.Tensor,
    direction: torch.Tensor | None,
    *,
    attribute: str,
    sample: int,
    strengths: Sequence[float],
    unit: float,
    applied: Sequence[Mapping[str, float]],
    weight_norm: float,
    weight_std: float,
    args: argparse.Namespace,
    debias_map: Mapping[str, Sequence[str]],
    diagnostics: dict[str, Any],
    fitted_samples: int,
    row_alignment: str,
) -> dict[str, Any]:
    ids = [
        f"wedit_{attribute}_s{sample}_a{strength:+.3f}" for strength in strengths
    ]
    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "x": tile_rows(original, len(strengths)),
        "x_hat": edited.to(original.dtype),
        "indices": ids,
        "ids": ids,
        "split": f"weight_edited_{attribute}",
        "epoch": metadata.get("epoch"),
        "config": metadata.get("config"),
        "source_file": str(args.weights.resolve()),
        "source_field": args.field,
        "space": "weight",
        "edit_attribute": attribute,
        "edit_sample_idx": sample,
        "edit_strengths": list(strengths),
        "edit_strength_mode": args.strength_mode,
        "edit_strength_unit": unit,
        "edit_applied": [dict(entry) for entry in applied],
        "edit_theta_norm": weight_norm,
        "edit_sigma_theta": weight_std,
        "edit_ridge_lambda": args.ridge_lambda,
        "edit_debias": list(debias_map.get(attribute, ())),
        "edit_n_fitted": fitted_samples,
        "edit_sign_convention": "positive alpha exaggerates the attribute",
        "edit_diagnostics": diagnostics,
        "row_alignment": row_alignment,
        "reference_note": (
            "x is the unedited adapter tiled over the sweep and is not ground "
            "truth for x_hat"
        ),
    }
    if direction is not None:
        payload["edit_direction"] = direction.cpu()
    return payload


def write_manifest(
    args: argparse.Namespace,
    *,
    source_shape: tuple[int, ...],
    fitted_samples: int,
    weight_std: float,
    strengths: Sequence[float],
    debias_map: Mapping[str, Sequence[str]],
    row_alignment: str,
    fit: DirectionFit,
    outputs: list[dict[str, Any]],
) -> Path:
    manifest = {
        "space": "weight",
        "source": str(args.weights.resolve()),
        "source_field": args.field,
        "source_shape": list(source_shape),
        "identity_df": str(args.identity_df.resolve()),
        "aligned_with": (
            str(args.align_with_latents.resolve()) if args.align_with_latents else None
        ),
        "row_alignment": row_alignment,
        "n_samples_fitted": fitted_samples,
        "weight_dim": int(source_shape[1]),
        "sigma_theta": weight_std,
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
    path = args.output_dir / "wedit_manifest.json"
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
    probes: dict[int, torch.Tensor] = {}
    limit = args.limit
    if args.align_with_latents is not None:
        print(f"Reading the fitted population size from {args.align_with_latents}")
        probe_rows = (
            sorted(set(rows) | set(range(args.verify_row_count)))
            if args.verify_alignment
            else ()
        )
        limit, probes = read_alignment_probes(
            args.align_with_latents,
            probe_rows,
            mmap=args.mmap,
        )
        print(f"  N = {limit} (first {limit} adapter rows)")
    elif limit is None:
        print(
            "No --align-with-latents given; fitting on the whole population "
            "unverified. Confirm it matches the population edit_latents.py fitted on."
        )

    print(f"Loading {args.weights}")
    weights, metadata = load_weight_population(
        args.weights,
        field=args.field,
        limit=limit,
        mmap=args.mmap,
    )
    sample_count, width = weights.shape
    print(
        f"  using: {tuple(weights.shape)} "
        f"({weights.element_size() * weights.nelement() / 1e9:.2f} GB) on {device}"
    )

    invalid = [row for row in rows if row < 0 or row >= sample_count]
    if invalid:
        raise IndexError(f"--rows {invalid} are outside 0..{sample_count - 1}")
    if args.verify_alignment and args.align_with_latents is not None:
        print("\nVerifying that latent row i and weight row i are the same adapter:")
        row_alignment = verify_alignment(weights, probes, args.verify_tolerance)
    else:
        row_alignment = "first N rows"
    if args.expected_weight_dim is not None and width != args.expected_weight_dim:
        raise ValueError(
            f"Adapter dimension is {width} but --expected-weight-dim is "
            f"{args.expected_weight_dim}. experiments/generate_images.py asserts the "
            "shape, so this would fail there instead (rank-1: 99648, rank-2: 199296)."
        )

    # The unit in which weight-space robustness was measured.
    weight_std = float(weights.std().item())
    print(f"  sigma_theta (global weight s.d.) = {weight_std:.6f}")

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

    edit_rows = {row: weights[row].clone() for row in rows}
    # Centre in place; the rows to edit are already copied out.
    weights -= weights.mean(dim=0, keepdim=True)

    fit = fit_directions(
        weights,
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
    print(
        "\n  Compare these AUCs against the latent ones before rendering anything.\n"
        "  Similar AUC means the attribute is linearly encoded just as well in the\n"
        "  raw weights, so any difference in the rendered edit is about what the two\n"
        "  spaces do when you MOVE, not about what they encode."
    )
    del weights

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for sample in rows:
        original = edit_rows[sample].to(device)
        weight_norm = float(original.norm().item())
        print(f"\nSample {sample} (||theta|| = {weight_norm:.3f})")

        for attribute in targets:
            diagnostics = fit.diagnostics[attribute]
            direction = fit.directions[attribute]
            unit = strength_unit(args.strength_mode, diagnostics, weight_norm)
            applied = applied_strengths(
                strengths,
                unit=unit,
                weight_norm=weight_norm,
                weight_std=weight_std,
                width=width,
            )
            print(
                f"  {attribute:<20} AUC {diagnostics['auc_holdout_mean']:.3f}   "
                f"unit ({args.strength_mode}) = {unit:.4f}"
            )
            worst = report_applied_strengths(
                applied,
                safe_scale=args.safe_noise_scale,
                collapse_scale=args.collapse_noise_scale,
            )

            edited = torch.stack(
                [
                    (original + strength * unit * direction).cpu()
                    for strength in strengths
                ],
                dim=0,
            )
            output_path = planned[(attribute, sample)]
            torch.save(
                build_edit_payload(
                    metadata,
                    original.cpu(),
                    edited,
                    direction if args.save_directions else None,
                    attribute=attribute,
                    sample=sample,
                    strengths=strengths,
                    unit=unit,
                    applied=applied,
                    weight_norm=weight_norm,
                    weight_std=weight_std,
                    args=args,
                    debias_map=debias_map,
                    diagnostics=diagnostics,
                    fitted_samples=sample_count,
                    row_alignment=row_alignment,
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
                    "auc_holdout": diagnostics["auc_holdout_mean"],
                    "strength_unit": unit,
                    "max_noise_equivalent_a": worst,
                    "output": str(output_path.resolve()),
                }
            )
            print(f"    -> {output_path.name} ({len(strengths)} rows)")

    manifest_path = write_manifest(
        args,
        source_shape=(sample_count, width),
        fitted_samples=sample_count,
        weight_std=weight_std,
        strengths=strengths,
        debias_map=debias_map,
        row_alignment=row_alignment,
        fit=fit,
        outputs=outputs,
    )
    print(f"\nSaved {len(outputs)} weight edit sweep(s)")
    print(f"Manifest: {manifest_path}")
    print(f"Row order within every file = strengths {strengths}")
    if 0.0 in strengths:
        print(
            f"Row {strengths.index(0.0)} (alpha = 0) is the unmodified adapter. Unlike "
            "the latent control it is not a reconstruction, so it should render "
            "identically to the original; if it does not, something upstream is wrong."
        )
    print(
        "Next: render x_hat directly with experiments/generate_images.py "
        "(--fields x_hat --no-scores); there is no decode stage in weight space."
    )


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

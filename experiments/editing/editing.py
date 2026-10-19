"""Shared attribute-direction fitting for matched latent- and weight-space edits.

Both editing scripts fit a direction for every requested attribute by ridge
regression against its binary labels, optionally remove the component that
aligns with a nuisance attribute, and report held-out AUC before any edit is
applied. Everything runs in the ``N x N`` kernel: the Gram matrix is built once,
all attributes are solved as a single multi-RHS system, and debiasing is done in
the dual, so no ``D``-dimensional vector exists until the final materialization
step.

With ``w = Zc^T a`` and ``a = (Zc Zc^T + lambda I)^-1 y``, the projections are
``pi = K a`` and ``||w||^2 = a^T K a``, so the shared Gram matrix
``K = Zc Zc^T`` is all that either script ever needs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_RIDGE_LAMBDA = 1.0
DEFAULT_HOLDOUT_FRACTION = 0.3
DEFAULT_HOLDOUT_SPLITS = 5
DEFAULT_HOLDOUT_SEED = 2020
DEFAULT_AUC_WARNING = 0.75
DEFAULT_GRAM_CHUNK = 512
DEFAULT_S_THRESHOLD = 0.2


@dataclass(frozen=True)
class DirectionFit:
    """Unit attribute directions in the input space plus their diagnostics."""

    attributes: list[str]
    directions: dict[str, torch.Tensor]
    diagnostics: dict[str, dict[str, Any]]


def load_identity_frame(path: Path) -> Any:
    """Load the identity dataframe (a pickled pandas DataFrame via torch.load)."""

    if not path.is_file():
        raise FileNotFoundError(f"Identity dataframe not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def label_vector(frame: Any, attribute: str, sample_count: int) -> torch.Tensor:
    """Return one attribute column as a ``[sample_count]`` float tensor of 0/1.

    Row ``i`` of the dataframe corresponds to sample row ``i``, so this is a
    plain truncation. CelebA attributes are stored as -1/+1; any such encoding is
    normalised so that a positive value means the attribute is present.
    """

    if attribute not in frame.columns:
        raise ValueError(
            f"Attribute {attribute!r} not found in the identity dataframe.\n"
            f"Available: {sorted(frame.columns)}"
        )
    if len(frame) < sample_count:
        raise ValueError(
            f"Identity dataframe has {len(frame)} rows for {sample_count} samples"
        )
    values = torch.tensor(
        frame.iloc[:sample_count][attribute].to_numpy(),
        dtype=torch.float32,
    )
    if values.min() < 0:
        return (values > 0).float()
    return values


def parse_debias_map(specs: Sequence[str] | None) -> dict[str, list[str]]:
    """Parse repeated ``TARGET=NUISANCE[,NUISANCE...]`` options into a mapping."""

    debias_map: dict[str, list[str]] = {}
    for spec in specs or ():
        target, separator, nuisances = spec.partition("=")
        target = target.strip()
        if not separator or not target:
            raise ValueError(
                f"Invalid --debias value {spec!r}; expected TARGET=NUISANCE[,NUISANCE...]"
            )
        selected = [name.strip() for name in nuisances.split(",") if name.strip()]
        if not selected:
            raise ValueError(f"--debias entry {spec!r} lists no nuisance attributes")
        if target in selected:
            raise ValueError(
                f"{target!r} appears in its own --debias entry. Projecting a "
                "direction onto itself and subtracting leaves a zero vector, which "
                "is then renormalised into noise."
            )
        existing = debias_map.setdefault(target, [])
        existing.extend(name for name in selected if name not in existing)
    return debias_map


def resolve_labels(
    frame: Any,
    targets: Sequence[str],
    debias_map: Mapping[str, Sequence[str]],
    sample_count: int,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Collect the label vectors for every target and nuisance attribute."""

    if not targets:
        raise ValueError("At least one attribute must be selected")
    unused = [target for target in debias_map if target not in targets]
    if unused:
        raise ValueError(f"--debias names attributes that are not edited: {unused}")

    nuisances = [name for values in debias_map.values() for name in values]
    needed = list(dict.fromkeys([*targets, *nuisances]))
    labels: dict[str, torch.Tensor] = {}
    for attribute in needed:
        values = label_vector(frame, attribute, sample_count)
        if values.std() == 0:
            raise ValueError(
                f"{attribute!r} is constant over the {sample_count} fitted samples "
                "(no positive/negative contrast). Remove it."
            )
        labels[attribute] = values
    return needed, labels


def find_binary_attributes(
    frame: Any,
    sample_count: int,
) -> tuple[list[str], list[str], list[str]]:
    """Auto-detect the binary attribute columns of an identity dataframe.

    A column qualifies if, over the first ``sample_count`` rows, it is numeric
    and its unique values are a subset of {-1, 0, 1}. Constant columns carry no
    signal and are reported separately rather than silently screened.
    """

    usable, constant, skipped = [], [], []
    rows = frame.iloc[:sample_count]
    for column in frame.columns:
        try:
            values = torch.tensor(rows[column].to_numpy(), dtype=torch.float32)
        except (TypeError, ValueError):
            skipped.append(column)
            continue
        if not torch.isfinite(values).all():
            skipped.append(column)
            continue
        unique = torch.unique(values)
        if unique.numel() > 3 or not bool(
            ((unique == -1) | (unique == 0) | (unique == 1)).all()
        ):
            skipped.append(column)
            continue
        if unique.numel() < 2:
            constant.append(column)
            continue
        usable.append(column)
    return usable, constant, skipped


def label_matrix(
    frame: Any,
    attributes: Sequence[str],
    sample_count: int,
) -> torch.Tensor:
    """Stack the attribute columns into ``Y`` with shape ``[sample_count, A]``."""

    return torch.stack(
        [label_vector(frame, attribute, sample_count) for attribute in attributes],
        dim=1,
    )


def projections_from_alpha(
    gram_cross: torch.Tensor,
    alpha: torch.Tensor,
    gram_fit: torch.Tensor,
) -> torch.Tensor:
    """Project samples onto the unit directions for every attribute at once.

    ``pi = Z_eval w = K_cross a`` and ``||w||^2 = a^T K_fit a``, so neither the
    directions nor the evaluation latents leave the kernel.
    """

    return (gram_cross @ alpha) / (
        alpha_norm(gram_fit, alpha).unsqueeze(0) + 1e-12
    )


def separation_ratios(
    projections: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-attribute ``s``, separation, and spread for ``[m, A]`` projections.

    ``s`` is invariant to the scale of the direction, so the unit normalisation
    matters for the reported mean projections but not for ``s`` itself.
    """

    positive = labels > 0.5
    negative = ~positive
    positive_count = positive.sum(dim=0).clamp(min=1)
    negative_count = negative.sum(dim=0).clamp(min=1)
    separation = (projections * positive).sum(dim=0) / positive_count - (
        projections * negative
    ).sum(dim=0) / negative_count
    spread = projections.std(dim=0)
    return separation / (spread + 1e-12), separation, spread


def auc_scores(
    projections: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Rank-based AUC per attribute for ``[m, A]`` projections and labels."""

    count, attribute_count = projections.shape
    order = projections.argsort(dim=0)
    ranks = torch.empty_like(order, dtype=torch.float64)
    positions = torch.arange(
        1,
        count + 1,
        dtype=torch.float64,
        device=projections.device,
    ).unsqueeze(1)
    ranks.scatter_(0, order, positions.expand(count, attribute_count))

    positive = labels > 0.5
    positive_count = positive.sum(dim=0).double()
    negative_count = (~positive).sum(dim=0).double()
    rank_sum = (ranks * positive).sum(dim=0)
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    ).clamp(min=1)


def direction_cosines(
    gram: torch.Tensor,
    alphas: torch.Tensor,
) -> torch.Tensor:
    """Pairwise cosine similarity of fitted directions, computed in the kernel.

    ``<w_i, w_j> = a_i^T K a_j``, so no ``D``-dimensional vector is formed.
    """

    inner = alphas.T @ (gram @ alphas)
    norms = torch.sqrt(torch.clamp(inner.diagonal(), min=1e-12))
    return inner / (norms.unsqueeze(0) * norms.unsqueeze(1))


def gram_matrix(
    centered: torch.Tensor,
    device: torch.device,
    chunk: int,
) -> torch.Tensor:
    """``K = Zc Zc^T`` accumulated chunk-wise, so only ``chunk`` rows are resident."""

    sample_count = centered.shape[0]
    gram = torch.empty(
        (sample_count, sample_count),
        dtype=torch.float64,
        device=device,
    )
    for start in range(0, sample_count, chunk):
        stop = min(start + chunk, sample_count)
        block = centered[start:stop].to(device)
        for other_start in range(0, sample_count, chunk):
            other_stop = min(other_start + chunk, sample_count)
            other = centered[other_start:other_stop].to(device)
            gram[start:stop, other_start:other_stop] = (block @ other.T).double()
    return 0.5 * (gram + gram.T)  # kill float32 accumulation asymmetry


def ridge_dual_multi(
    gram: torch.Tensor,
    centered_labels: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    """``a = (K + lambda I)^-1 Yc``, solved for every attribute column at once."""

    system = gram.clone()
    system.diagonal().add_(ridge_lambda)
    return torch.linalg.solve(system, centered_labels.double())


def alpha_norm(gram: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """``||w||`` for ``w = Zc^T a`` without forming ``w``: ``||w||^2 = a^T K a``."""

    return torch.sqrt(torch.clamp((alpha * (gram @ alpha)).sum(dim=0), min=0.0))


def debias_alpha(
    gram: torch.Tensor,
    target: torch.Tensor,
    nuisance: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Gram-Schmidt removal performed in the dual.

    With ``w = Zc^T a``, the coefficient ``c = <w_t, w_b> / ||w_b||^2`` is
    ``(a_t^T K a_b) / (a_b^T K a_b)``, and ``w_t - c w_b = Zc^T (a_t - c a_b)``,
    so the debiased direction is just another alpha.
    """

    projected = gram @ nuisance
    denominator = (nuisance * projected).sum()
    coefficient = (target * projected).sum() / (denominator + 1e-30)
    return target - coefficient * nuisance, float(coefficient.item())


def auc_from_projection(
    projections: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Rank-based AUC (Mann-Whitney U) of the projections against the labels.

    Reported in place of the separation ratio because it has no mechanical
    dependence on class balance, whereas the in-sample separation ratio of an
    interpolating fit is capped at ``1/sqrt(p(1-p))``.
    """

    count = projections.numel()
    order = projections.argsort()
    ranks = torch.empty(count, dtype=torch.float64, device=projections.device)
    ranks[order] = torch.arange(
        1,
        count + 1,
        dtype=torch.float64,
        device=projections.device,
    )
    positive = labels > 0.5
    positive_count = float(positive.sum())
    negative_count = float((~positive).sum())
    if positive_count < 1 or negative_count < 1:
        return float("nan")
    rank_sum = ranks[positive].sum() - positive_count * (positive_count + 1) / 2
    return float((rank_sum / (positive_count * negative_count)).item())


def separation_stats(
    projections: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Separation, spread, their ratio, and the ceiling implied by prevalence."""

    positive = labels > 0.5
    if positive.sum() < 1 or (~positive).sum() < 1:
        return float("nan"), float("nan"), float("nan"), float("nan")
    separation = float((projections[positive].mean() - projections[~positive].mean()).item())
    spread = float(projections.std().item())
    prevalence = float(positive.float().mean().item())
    ceiling = 1.0 / math.sqrt(prevalence * (1 - prevalence))
    return separation, spread, separation / (spread + 1e-12), ceiling


def holdout_auc(
    gram: torch.Tensor,
    label_matrix: torch.Tensor,
    targets: Sequence[str],
    index_of: Mapping[str, int],
    debias_map: Mapping[str, Sequence[str]],
    *,
    ridge_lambda: float,
    holdout_fraction: float,
    splits: int,
    seed: int,
) -> dict[str, list[float]]:
    """Repeated random-split held-out AUC for every target direction.

    Fitted on all ``N`` samples the in-sample separation ratio equals
    ``1/sqrt(p(1-p))`` exactly, so it carries no information about the attribute.
    These splits give an honest number; the direction that is actually applied is
    still the all-data fit.
    """

    sample_count = gram.shape[0]
    holdout_count = max(1, int(round(holdout_fraction * sample_count)))
    fit_count = sample_count - holdout_count
    if fit_count < 2:
        raise ValueError(
            f"--holdout-fraction {holdout_fraction} leaves {fit_count} fitting rows"
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scores: dict[str, list[float]] = {attribute: [] for attribute in targets}

    for _ in range(splits):
        permutation = torch.randperm(sample_count, generator=generator).to(gram.device)
        fit_rows, eval_rows = permutation[:fit_count], permutation[fit_count:]
        gram_fit = gram[fit_rows][:, fit_rows]
        gram_eval = gram[eval_rows][:, fit_rows]
        fit_labels = label_matrix[fit_rows]
        alphas = ridge_dual_multi(
            gram_fit,
            fit_labels - fit_labels.mean(dim=0, keepdim=True),
            ridge_lambda,
        )
        for attribute in targets:
            alpha = alphas[:, index_of[attribute]].clone()
            for nuisance in debias_map.get(attribute, ()):
                alpha, _ = debias_alpha(gram_fit, alpha, alphas[:, index_of[nuisance]])
            norm = alpha_norm(gram_fit, alpha.unsqueeze(1)).squeeze()
            projections = (gram_eval @ alpha) / (norm + 1e-12)
            scores[attribute].append(
                auc_from_projection(
                    projections,
                    label_matrix[eval_rows][:, index_of[attribute]],
                )
            )
    return scores


def materialize_directions(
    centered: torch.Tensor,
    alphas: torch.Tensor,
    device: torch.device,
    chunk: int,
) -> torch.Tensor:
    """Form ``W = Zc^T A`` and renormalise, the only ``D``-dimensional step."""

    sample_count, width = centered.shape
    stacked = torch.zeros(
        (width, alphas.shape[1]),
        dtype=torch.float64,
        device=device,
    )
    for start in range(0, sample_count, chunk):
        stop = min(start + chunk, sample_count)
        stacked += centered[start:stop].to(device).double().T @ alphas[start:stop]
    materialized = stacked.float()
    materialized /= materialized.norm(dim=0, keepdim=True) + 1e-12
    return materialized


def fit_directions(
    centered: torch.Tensor,
    labels: Mapping[str, torch.Tensor],
    targets: Sequence[str],
    fitted_attributes: Sequence[str],
    *,
    debias_map: Mapping[str, Sequence[str]],
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    splits: int = DEFAULT_HOLDOUT_SPLITS,
    seed: int = DEFAULT_HOLDOUT_SEED,
    orient_positive_adds: bool = True,
    device: torch.device,
    chunk: int = DEFAULT_GRAM_CHUNK,
) -> DirectionFit:
    """Fit, debias, diagnose, and materialize one unit direction per target.

    ``centered`` must already be centred on the fitting population; the caller
    keeps the uncentred rows it intends to edit.
    """

    sample_count = centered.shape[0]
    print("Building Gram matrix (the only O(N^2 D) step) ...")
    gram = gram_matrix(centered, device, chunk)
    print(
        f"  K: {tuple(gram.shape)} "
        f"({gram.element_size() * gram.nelement() / 1e6:.0f} MB)"
    )

    label_matrix = torch.stack(
        [labels[attribute] for attribute in fitted_attributes],
        dim=1,
    ).to(device)
    index_of = {attribute: index for index, attribute in enumerate(fitted_attributes)}
    print(f"Solving {len(fitted_attributes)} ridge systems (lambda={ridge_lambda}) ...")
    alphas = ridge_dual_multi(
        gram,
        label_matrix - label_matrix.mean(dim=0, keepdim=True),
        ridge_lambda,
    )

    print(
        f"Held-out diagnostics: {splits} splits, "
        f"{int((1 - holdout_fraction) * 100)}/{int(holdout_fraction * 100)} split"
    )
    auc_scores = holdout_auc(
        gram,
        label_matrix,
        targets,
        index_of,
        debias_map,
        ridge_lambda=ridge_lambda,
        holdout_fraction=holdout_fraction,
        splits=splits,
        seed=seed,
    )

    final_alphas: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for attribute in targets:
        alpha = alphas[:, index_of[attribute]].clone()
        removed = []
        for nuisance in debias_map.get(attribute, ()):
            alpha, coefficient = debias_alpha(
                gram,
                alpha,
                alphas[:, index_of[nuisance]],
            )
            removed.append({"attribute": nuisance, "coefficient": coefficient})

        norm = alpha_norm(gram, alpha.unsqueeze(1)).squeeze()
        if float(norm) < 1e-8:
            raise RuntimeError(
                f"Direction for {attribute!r} collapsed to zero after debiasing"
            )
        alpha = alpha / norm  # so that ||w|| == 1

        attribute_labels = labels[attribute].to(device)
        projections = gram @ alpha
        separation, spread, in_sample, ceiling = separation_stats(
            projections,
            attribute_labels,
        )
        if orient_positive_adds and separation < 0:
            alpha, separation = -alpha, -separation

        scores = auc_scores[attribute]
        auc_mean = sum(scores) / len(scores)
        auc_std = (sum((value - auc_mean) ** 2 for value in scores) / len(scores)) ** 0.5
        final_alphas[attribute] = alpha
        diagnostics[attribute] = {
            "prevalence": float(attribute_labels.mean().item()),
            "n_positive": int(attribute_labels.sum().item()),
            "auc_holdout_mean": auc_mean,
            "auc_holdout_std": auc_std,
            "auc_holdout_splits": scores,
            "separation": separation,
            "spread": spread,
            "s_in_sample": in_sample,
            "s_ceiling": ceiling,
            "debiased_against": removed,
        }

    print("Materializing directions ...")
    stacked = torch.stack([final_alphas[attribute] for attribute in targets], dim=1)
    materialized = materialize_directions(centered, stacked, device, chunk)
    directions = {
        attribute: materialized[:, index]
        for index, attribute in enumerate(targets)
    }
    print(f"  fitted {len(directions)} direction(s) over {sample_count} samples")
    return DirectionFit(
        attributes=list(targets),
        directions=directions,
        diagnostics=diagnostics,
    )


def print_direction_table(
    fit: DirectionFit,
    *,
    auc_warning: float = DEFAULT_AUC_WARNING,
) -> None:
    """Print the per-attribute diagnostics table shared by both scripts."""

    print()
    header = (
        f"  {'attribute':<20}{'prev':>7}{'AUC(held-out)':>16}"
        f"{'sep':>12}{'spread':>10}{'s_in':>8}{'ceil':>8}  debias"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for attribute in fit.attributes:
        details = fit.diagnostics[attribute]
        auc = details["auc_holdout_mean"]
        removed = ",".join(entry["attribute"] for entry in details["debiased_against"])
        flag = "   <-- weak" if auc < auc_warning else ""
        print(
            f"  {attribute:<20}{details['prevalence'] * 100:6.2f}%"
            f"{auc:11.3f}+-{details['auc_holdout_std']:.3f}"
            f"{details['separation']:12.4f}{details['spread']:10.4f}"
            f"{details['s_in_sample']:8.2f}{details['s_ceiling']:8.2f}"
            f"  {removed or '-'}{flag}"
        )
    print(
        "\n  s_in matches the ceiling 1/sqrt(p(1-p)) for every attribute: the "
        "all-data ridge\n  fit interpolates the labels, so only the held-out AUC "
        "is informative."
    )


def attribute_group(
    attribute: str,
    targets: Sequence[str],
    controls: Sequence[str],
) -> str:
    """Label an attribute as a target or as a predicted-failure control."""

    if attribute in controls:
        return "control"
    if attribute in targets:
        return "target"
    return "other"


def tile_rows(tensor: torch.Tensor, count: int) -> torch.Tensor:
    """Repeat a single sample row ``count`` times along a new leading axis."""

    return tensor.unsqueeze(0).repeat(count, *([1] * tensor.dim()))


def unique_attributes(*groups: Iterable[str]) -> list[str]:
    """Concatenate attribute groups, preserving order and dropping duplicates."""

    return list(dict.fromkeys(name for group in groups for name in group))

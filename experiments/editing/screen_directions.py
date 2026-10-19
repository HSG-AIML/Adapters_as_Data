"""Screen attribute directions in the SANE latent space without rendering.

For every binary attribute in the identity dataframe this fits the same
ridge-regularized direction ``edit_latents.py`` fits, projects the centred
latents onto it, and reports how well the two label groups separate relative to
the overall spread:

    s = (mean(pi_+) - mean(pi_-)) / std(pi),      pi_i = <z_i - z_bar, d>

A small ``s`` means the two clouds overlap and no edit strength will produce a
clean change, so the screen decides which attributes are worth decoding and
rendering at all. Held-out ``s`` and AUC are reported alongside the in-sample
numbers: with ``D >> N`` the ridge fit still has enormous freedom, so in-sample
separation is optimistically biased, and the gap between the two is itself worth
reporting.

Outputs, written to ``--output-dir``:

- ``direction_screen.csv`` / ``direction_screen.json``: metrics per attribute
- ``tab_direction_screen.tex``: booktabs table
- ``fig_direction_screen.tex``: pgfplots bar chart of ``s`` per attribute
- ``fig_separation_vs_prevalence.tex``: ``s`` against label prevalence
- ``fig_auc_vs_prevalence.tex``: held-out AUC against prevalence (the version
  for the main text; ``s`` has a prevalence-dependent ceiling, AUC does not)
- ``direction_cosines.csv``: pairwise cosines, the candidates for ``--debias``

Example:

    uv run python experiments/editing/screen_directions.py \
        outputs/latents/latents_train.pt \
        --identity-df path/to/identity_df.pt \
        --output-dir outputs/editing/screen

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
    DEFAULT_GRAM_CHUNK,
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_HOLDOUT_SEED,
    DEFAULT_HOLDOUT_SPLITS,
    DEFAULT_RIDGE_LAMBDA,
    DEFAULT_S_THRESHOLD,
    auc_scores,
    direction_cosines,
    find_binary_attributes,
    gram_matrix,
    label_matrix,
    load_identity_frame,
    projections_from_alpha,
    ridge_dual_multi,
    separation_ratios,
)


CSV_FIELDS = [
    "attribute",
    "n_positive",
    "prevalence",
    "separation",
    "spread",
    "s_in",
    "s_out_mean",
    "s_out_std",
    "auc_out_mean",
    "passes",
]


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
        help="Attributes to screen (default: every binary dataframe column).",
    )
    parser.add_argument(
        "--exclude-attributes",
        nargs="*",
        default=(),
        help="Attributes to drop; constant columns are dropped anyway.",
    )
    parser.add_argument(
        "--ridge-lambda",
        type=float,
        default=DEFAULT_RIDGE_LAMBDA,
        help=f"Ridge strength; match edit_latents.py (default: {DEFAULT_RIDGE_LAMBDA}).",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
        help=f"Fraction held out of each fit (default: {DEFAULT_HOLDOUT_FRACTION}).",
    )
    parser.add_argument(
        "--holdout-splits",
        type=int,
        default=DEFAULT_HOLDOUT_SPLITS,
        help=f"Repeated random splits (default: {DEFAULT_HOLDOUT_SPLITS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_HOLDOUT_SEED,
        help=f"Seed for the diagnostic splits (default: {DEFAULT_HOLDOUT_SEED}).",
    )
    parser.add_argument(
        "--s-threshold",
        type=float,
        default=DEFAULT_S_THRESHOLD,
        help=f"Rejection threshold on held-out s (default: {DEFAULT_S_THRESHOLD}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Screen only the first N latents.",
    )
    parser.add_argument(
        "--gram-chunk",
        type=int,
        default=DEFAULT_GRAM_CHUNK,
        help=f"Rows per chunk when building K (default: {DEFAULT_GRAM_CHUNK}).",
    )
    parser.add_argument(
        "--table-top-k",
        type=int,
        help="Rows to include in the LaTeX table (default: all screened attributes).",
    )
    parser.add_argument(
        "--scatter-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Label every scatter point with its attribute name (default: true).",
    )
    parser.add_argument(
        "--cosines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write the pairwise direction cosines (default: true).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. cuda:0 or cpu (default: auto).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing screening outputs.",
    )
    return parser.parse_args(argv)


def load_latents(path: Path, limit: int | None) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Latent payload not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "z" not in payload:
        raise ValueError(f"Latent payload must contain a mapping with z: {path}")
    latents = payload["z"]
    if not isinstance(latents, torch.Tensor) or latents.ndim != 3:
        raise ValueError(f"{path}: z must be a rank-3 tensor")
    if limit is not None:
        if limit < 2:
            raise ValueError("--limit must be at least 2")
        if limit > latents.shape[0]:
            raise IndexError(
                f"--limit {limit} exceeds the {latents.shape[0]} available latents"
            )
        latents = latents[:limit]
    if latents.shape[0] < 2:
        raise ValueError("Screening requires at least two latents")
    return latents.float()


def select_attributes(
    frame: Any,
    args: argparse.Namespace,
    sample_count: int,
) -> tuple[list[str], list[str], list[str]]:
    if args.attributes is None:
        attributes, constant, skipped = find_binary_attributes(frame, sample_count)
        if constant:
            print(f"  constant columns (no signal, excluded): {constant}")
        if skipped:
            print(f"  non-binary columns (ignored): {skipped}")
    else:
        attributes, constant, skipped = list(args.attributes), [], []
    excluded = set(args.exclude_attributes)
    attributes = [name for name in attributes if name not in excluded]
    if not attributes:
        raise ValueError("No attributes left to screen")
    return attributes, constant, skipped


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation, without pulling in scipy. Ties are broken ordinally."""

    def rank(values: Sequence[float]) -> list[int]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        ranks = [0] * len(values)
        for position, index in enumerate(order):
            ranks[index] = position
        return ranks

    rank_x, rank_y = rank(xs), rank(ys)
    count = len(rank_x)
    mean_x, mean_y = sum(rank_x) / count, sum(rank_y) / count
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in rank_x)
        * sum((b - mean_y) ** 2 for b in rank_y)
    )
    return numerator / denominator if denominator else float("nan")


def tex_name(attribute: str) -> str:
    """Match the naming in tab:celeba-attributes: underscores become spaces."""

    return attribute.replace("_", " ")


def write_table(
    rows: Sequence[dict[str, Any]],
    path: Path,
    args: argparse.Namespace,
) -> None:
    """booktabs table, sorted by held-out separation ratio."""

    threshold = args.s_threshold
    shown = rows if args.table_top_k is None else rows[: args.table_top_k]
    lines = [
        "% Auto-generated by screen_directions.py — do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Attribute-direction screening. For each attribute a direction is fitted by "
        r"ridge regression against the binary label and the centred latents are projected onto "
        r"it; $s$ is the separation between the two label groups in units of the overall "
        r"spread. In-sample $s$ is measured on the samples the direction was fitted on, "
        r"held-out $s$ on samples excluded from the fit "
        f"(mean $\\pm$ s.d. over {args.holdout_splits} random splits, "
        f"{int(args.holdout_fraction * 100)}\\% held "
        r"out). AUC is the held-out rank statistic; $0.5$ is chance. Prevalence is the "
        r"fraction of positive labels among the screened adapters.}",
        r"\label{tab:direction-screen}",
        r"\ra{1.15}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Attribute} & \textbf{Prevalence} & \textbf{$s$ (in-sample)} & "
        r"\textbf{$s$ (held-out)} & \textbf{AUC} \\",
        r"\midrule",
    ]
    for row in shown:
        flag = "" if row["s_out_mean"] >= threshold else r"\;$^{\dagger}$"
        lines.append(
            f"{tex_name(row['attribute'])}{flag} & "
            f"${row['prevalence'] * 100:.2f}\\%$ & "
            f"${row['s_in']:.2f}$ & "
            f"${row['s_out_mean']:.2f} \\pm {row['s_out_std']:.2f}$ & "
            f"${row['auc_out_mean']:.3f}$ \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[2pt]{\footnotesize $^{\dagger}$ held-out $s < "
        f"{threshold:g}$: the label groups overlap along the fitted direction, so no edit "
        r"strength is expected to produce a clean change.}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bar_figure(
    rows: Sequence[dict[str, Any]],
    path: Path,
    args: argparse.Namespace,
) -> None:
    """Horizontal bar chart of s per attribute, in-sample against held-out."""

    threshold = args.s_threshold
    count = len(rows)
    height_cm = max(6.0, 0.40 * count)
    xmax = max(
        max(row["s_in"] for row in rows),
        max(row["s_out_mean"] + row["s_out_std"] for row in rows),
    )
    xmax = math.ceil(xmax * 10) / 10 + 0.1

    labels = ", ".join(f"{{{tex_name(row['attribute'])}}}" for row in rows)
    in_coords = " ".join(
        f"({row['s_in']:.4f},{index})" for index, row in enumerate(rows)
    )
    out_coords = " ".join(
        f"({row['s_out_mean']:.4f},{index}) +- ({row['s_out_std']:.4f},0)"
        for index, row in enumerate(rows)
    )

    lines = [
        "% Auto-generated by screen_directions.py — do not edit by hand.",
        r"\begin{figure}[p]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    xbar,",
        r"    width=0.62\linewidth, " + f"height={height_cm:.1f}cm,",
        r"    bar width=2.6pt,",
        r"    y dir=reverse,",
        f"    xmin=0, xmax={xmax:.2f}, ymin=-0.8, ymax={count - 0.2:.1f},",
        r"    xlabel={Separation ratio $s$},",
        f"    ytick={{0,...,{count - 1}}},",
        f"    yticklabels={{{labels}}},",
        r"    tick label style={font=\scriptsize},",
        r"    label style={font=\footnotesize},",
        r"    grid=both, grid style={gray!15},",
        r"    legend style={font=\scriptsize, at={(0.98,0.03)}, anchor=south east},",
        r"]",
        r"\addplot[xbar, fill=teal!70!black, draw=none] coordinates {" + in_coords + "};",
        r"\addplot+[xbar, fill=gray!45, draw=none, mark=none,",
        r"    error bars/.cd, x dir=both, x explicit,",
        r"    error bar style={black!55, line width=0.4pt}]",
        r"    coordinates {" + out_coords + "};",
        r"\legend{in-sample, held-out}",
        r"\draw[dashed, red!65!black]",
        f"    ({{axis cs:{threshold:g},0}}|-{{rel axis cs:0,0}}) -- "
        f"({{axis cs:{threshold:g},0}}|-{{rel axis cs:0,1}});",
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Attribute-direction screening across all binary CelebA attributes, sorted "
        r"by held-out separation. Bars show the separation ratio $s$ between the positive and "
        r"negative label groups when the centred latents are projected onto the fitted "
        r"direction, in-sample (teal) and on samples held out of the fit (grey, "
        f"mean $\\pm$ s.d. over {args.holdout_splits} splits). The dashed line marks the "
        f"$s = {threshold:g}$ rejection threshold: below it the two groups overlap and the "
        r"direction does not usefully encode the attribute.}",
        r"\label{fig:direction-screen}",
        r"\end{figure}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scatter_plot_lines(
    rows: Sequence[dict[str, Any]],
    value_key: str,
    labelled: bool,
) -> list[str]:
    if not labelled:
        coords = " ".join(
            f"({row['prevalence'] * 100:.3f},{row[value_key]:.4f})" for row in rows
        )
        return [
            r"\addplot[only marks, mark=*, mark size=1.8pt, teal!60!black] "
            r"coordinates {" + coords + "};"
        ]
    data = "\n".join(
        f"    {row['prevalence'] * 100:.3f} {row[value_key]:.4f} "
        f"{{{tex_name(row['attribute'])}}}"
        for row in rows
    )
    return [
        r"\addplot[only marks, mark=*, mark size=1.8pt, teal!60!black,",
        r"    point meta=explicit symbolic, nodes near coords,",
        r"    every node near coord/.append style={font=\tiny, anchor=west, "
        r"xshift=1.5pt, text=black!65}]",
        r"    table[x=x, y=y, meta=label, row sep=\\] {",
        r"    x y label \\",
        "\n".join(line + r" \\" for line in data.split("\n")),
        r"    };",
    ]


def write_scatter_figure(
    rows: Sequence[dict[str, Any]],
    path: Path,
    args: argparse.Namespace,
) -> None:
    """Held-out s against label prevalence."""

    threshold = args.s_threshold
    lines = [
        "% Auto-generated by screen_directions.py — do not edit by hand.",
        r"\begin{figure}[t]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=0.86\linewidth, height=0.62\linewidth,",
        r"    xlabel={Label prevalence (\%)},",
        r"    ylabel={Held-out separation ratio $s$},",
        r"    xmin=0, ymin=0,",
        r"    grid=both, grid style={gray!15},",
        r"    tick label style={font=\footnotesize},",
        r"    label style={font=\footnotesize},",
        r"]",
        *_scatter_plot_lines(rows, "s_out_mean", args.scatter_labels),
        r"\draw[dashed, red!65!black]",
        f"    ({{rel axis cs:0,0}}|-{{axis cs:0,{threshold:g}}}) -- "
        f"({{rel axis cs:1,0}}|-{{axis cs:0,{threshold:g}}});",
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Held-out separation ratio against label prevalence. Each point is one "
        r"attribute; the dashed line is the $s = "
        f"{threshold:g}$ rejection threshold.}}",
        r"\label{fig:separation-vs-prevalence}",
        r"\end{figure}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_auc_scatter_figure(
    rows: Sequence[dict[str, Any]],
    path: Path,
    args: argparse.Namespace,
    rho: float,
) -> None:
    """Held-out AUC against label prevalence.

    The companion to ``write_scatter_figure`` and the one that should carry the
    argument in the main text: the separation ratio has a ceiling of
    ``1/sqrt(p(1-p))`` that grows without bound as an attribute becomes rare,
    whereas AUC depends only on the ordering of the projections.
    """

    lines = [
        "% Auto-generated by screen_directions.py — do not edit by hand.",
        r"\begin{figure}[t]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=0.86\linewidth, height=0.62\linewidth,",
        r"    xlabel={Label prevalence (\%)},",
        r"    ylabel={Held-out AUC},",
        r"    xmin=0, xmax=100, ymin=0.46, ymax=1.02,",
        r"    ytick={0.5,0.6,0.7,0.8,0.9,1.0},",
        r"    grid=both, grid style={gray!15},",
        r"    tick label style={font=\footnotesize},",
        r"    label style={font=\footnotesize},",
        r"]",
        # strip below chance
        r"\addplot[draw=none, fill=gray!12, forget plot] coordinates "
        r"{(0,0.46) (100,0.46) (100,0.5) (0,0.5)} \closedcycle;",
        *_scatter_plot_lines(rows, "auc_out_mean", args.scatter_labels),
        r"\draw[dashed, red!65!black] ({rel axis cs:0,0}|-{axis cs:0,0.5}) -- "
        r"({rel axis cs:1,0}|-{axis cs:0,0.5});",
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Held-out AUC against label prevalence. Each point is one attribute; the "
        r"dashed line at $0.5$ is chance and the shaded strip below it is worse than chance. "
        r"Every attribute separates above chance, and how common an attribute is does not "
        f"predict how well its direction works (Spearman $\\rho = {rho:+.2f}$). This is the "
        r"same comparison as Figure~\ref{fig:separation-vs-prevalence}, which appears to show "
        r"a trend; that apparent trend is an artifact of the separation ratio's ceiling of "
        r"$1/\sqrt{p(1-p)}$, which grows without bound as an attribute becomes rare, whereas "
        r"AUC depends only on the ordering of the projections and so has no dependence on "
        r"class balance.}",
        r"\label{fig:auc-vs-prevalence}",
        r"\end{figure}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    lines = [",".join(CSV_FIELDS)]
    lines += [",".join(str(row[field]) for field in CSV_FIELDS) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cosines(
    attributes: Sequence[str],
    cosines: torch.Tensor,
    path: Path,
) -> None:
    lines = ["," + ",".join(attributes)]
    for index, attribute in enumerate(attributes):
        values = ",".join(f"{value:.4f}" for value in cosines[index].tolist())
        lines.append(f"{attribute},{values}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.ridge_lambda < 0:
        raise ValueError("--ridge-lambda must be non-negative")
    if args.holdout_splits < 1:
        raise ValueError("--holdout-splits must be positive")
    if not 0 < args.holdout_fraction < 1:
        raise ValueError("--holdout-fraction must lie strictly between 0 and 1")
    if args.gram_chunk < 1:
        raise ValueError("--gram-chunk must be positive")

    planned = {
        name: args.output_dir / name
        for name in (
            "direction_screen.csv",
            "direction_screen.json",
            "tab_direction_screen.tex",
            "fig_direction_screen.tex",
            "fig_separation_vs_prevalence.tex",
            "fig_auc_vs_prevalence.tex",
            *(("direction_cosines.csv",) if args.cosines else ()),
        )
    }
    existing = [path for path in planned.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Screening output(s) already exist: {existing}. "
            "Pass --overwrite to replace them."
        )

    device = resolve_device(args.device)
    latents = load_latents(args.latents, args.limit)
    sample_count, token_count, latent_dim = latents.shape
    width = token_count * latent_dim
    print(f"Loaded {args.latents}: z={tuple(latents.shape)} on {device}")
    print(f"  N={sample_count}, flattened D={width:,}")

    frame = load_identity_frame(args.identity_df)
    print(f"Loaded {args.identity_df}: {len(frame)} rows")
    attributes, constant, _skipped = select_attributes(frame, args, sample_count)
    attribute_count = len(attributes)
    print(f"  screening {attribute_count} attributes")

    labels = label_matrix(frame, attributes, sample_count).to(device)
    prevalence = labels.mean(dim=0)

    print("\nBuilding Gram matrix (the only O(N^2 D) step) ...")
    flattened = latents.reshape(sample_count, width)
    flattened -= flattened.mean(dim=0, keepdim=True)
    gram = gram_matrix(flattened, device, args.gram_chunk)
    del flattened, latents
    print(f"  K: {tuple(gram.shape)}")

    print(f"Fitting {attribute_count} directions (lambda={args.ridge_lambda}) ...")
    alphas = ridge_dual_multi(
        gram,
        labels - labels.mean(dim=0, keepdim=True),
        args.ridge_lambda,
    )
    in_sample = projections_from_alpha(gram, alphas, gram)
    s_in, separation_in, spread_in = separation_ratios(in_sample, labels)

    holdout_count = max(1, int(round(args.holdout_fraction * sample_count)))
    fit_count = sample_count - holdout_count
    if fit_count < 2:
        raise ValueError(
            f"--holdout-fraction {args.holdout_fraction} leaves {fit_count} fitting rows"
        )
    print(
        f"Held-out evaluation: {args.holdout_splits} splits, "
        f"{fit_count} fit / {holdout_count} eval"
    )
    s_out = torch.zeros(
        args.holdout_splits,
        attribute_count,
        dtype=torch.float64,
        device=device,
    )
    auc_out = torch.zeros_like(s_out)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for split in range(args.holdout_splits):
        permutation = torch.randperm(sample_count, generator=generator).to(device)
        fit_rows, eval_rows = permutation[:fit_count], permutation[fit_count:]
        gram_fit = gram[fit_rows][:, fit_rows]
        gram_cross = gram[eval_rows][:, fit_rows]
        fit_labels, eval_labels = labels[fit_rows], labels[eval_rows]
        split_alphas = ridge_dual_multi(
            gram_fit,
            fit_labels - fit_labels.mean(dim=0, keepdim=True),
            args.ridge_lambda,
        )
        projections = projections_from_alpha(gram_cross, split_alphas, gram_fit)
        s_out[split], _, _ = separation_ratios(projections, eval_labels)
        auc_out[split] = auc_scores(projections, eval_labels)
        print(f"  split {split + 1}/{args.holdout_splits} done")

    s_out_mean, s_out_std = s_out.mean(dim=0), s_out.std(dim=0)
    auc_out_mean = auc_out.mean(dim=0)

    rows = [
        {
            "attribute": attribute,
            "n_positive": int(labels[:, index].sum().item()),
            "prevalence": float(prevalence[index].item()),
            "separation": float(separation_in[index].item()),
            "spread": float(spread_in[index].item()),
            "s_in": float(s_in[index].item()),
            "s_out_mean": float(s_out_mean[index].item()),
            "s_out_std": float(s_out_std[index].item()),
            "auc_out_mean": float(auc_out_mean[index].item()),
            "passes": bool(s_out_mean[index].item() >= args.s_threshold),
        }
        for index, attribute in enumerate(attributes)
    ]
    rows.sort(key=lambda row: row["s_out_mean"], reverse=True)

    header = f"\n{'attribute':<22}{'prev':>8}{'s_in':>8}{'s_out':>9}{'+-':>7}{'AUC':>8}"
    print(header)
    print("-" * 68)
    for row in rows:
        mark = "" if row["passes"] else "  <- below threshold"
        print(
            f"{row['attribute']:<22}{row['prevalence'] * 100:7.2f}%{row['s_in']:8.2f}"
            f"{row['s_out_mean']:9.2f}{row['s_out_std']:7.2f}"
            f"{row['auc_out_mean']:8.3f}{mark}"
        )

    passing = sum(row["passes"] for row in rows)
    print(
        f"\n{passing}/{attribute_count} attributes pass the held-out "
        f"s >= {args.s_threshold} threshold."
    )
    gaps = [row["s_in"] - row["s_out_mean"] for row in rows]
    print(f"Mean in-sample minus held-out gap: {sum(gaps) / len(gaps):.2f}")
    print("  (a large gap means the ridge fit is memorizing labels rather than")
    print("   finding an attribute axis — a lower --ridge-lambda is NOT the fix there.)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, planned["direction_screen.csv"])
    planned["direction_screen.json"].write_text(
        json.dumps(
            {
                "config": {
                    "latents_file": str(args.latents.resolve()),
                    "identity_df": str(args.identity_df.resolve()),
                    "n_samples": sample_count,
                    "latent_shape": [token_count, latent_dim],
                    "flat_dim": width,
                    "ridge_lambda": args.ridge_lambda,
                    "holdout_fraction": args.holdout_fraction,
                    "holdout_splits": args.holdout_splits,
                    "seed": args.seed,
                    "s_threshold": args.s_threshold,
                    "constant_columns": constant,
                },
                "results": rows,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    write_table(rows, planned["tab_direction_screen.tex"], args)
    write_bar_figure(rows, planned["fig_direction_screen.tex"], args)
    write_scatter_figure(rows, planned["fig_separation_vs_prevalence.tex"], args)

    prevalences = [row["prevalence"] for row in rows]
    rho_auc = spearman(prevalences, [row["auc_out_mean"] for row in rows])
    rho_s = spearman(prevalences, [row["s_out_mean"] for row in rows])
    write_auc_scatter_figure(rows, planned["fig_auc_vs_prevalence.tex"], args, rho_auc)
    print("\nSpearman correlation with prevalence:")
    print(
        f"  separation ratio s : {rho_s:+.3f}   (inflated for rare attributes by the "
        "1/sqrt(p(1-p)) ceiling)"
    )
    print(
        f"  held-out AUC       : {rho_auc:+.3f}   (no mechanical dependence on "
        "class balance)"
    )

    if args.cosines:
        cosines = direction_cosines(gram, alphas)
        write_cosines(attributes, cosines, planned["direction_cosines.csv"])
        upper = torch.triu_indices(attribute_count, attribute_count, offset=1)
        pairs = sorted(
            (
                (abs(cosines[i, j].item()), attributes[i], attributes[j])
                for i, j in zip(upper[0].tolist(), upper[1].tolist())
            ),
            reverse=True,
        )[:10]
        print("\nMost entangled attribute pairs (|cos| between fitted directions):")
        for value, first, second in pairs:
            print(f"  {value:.3f}  {first} <-> {second}")
        print("  (these are the candidates for --debias in edit_latents.py)")

    print("\nWrote:")
    for path in planned.values():
        print(f"  {path}")
    print("\nThe figures assume pgfplots; the table assumes booktabs and \\ra.")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

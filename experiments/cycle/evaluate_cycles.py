"""Evaluate weight and image drift across SANE reconstruction cycles.

The input directory must be produced by ``experiments/cycle/cycle_latents.py``::

    cycle_root/
      round_00/x_and_x_hat.pt
      round_01/x_and_x_hat.pt
      ...

Every round payload intentionally stores the same original adapter in ``x`` and
the current reconstruction in ``x_hat``. Accordingly, cumulative metrics compare
``x_0`` with ``x_r``, while step metrics compare the previous round's ``x_hat``
with the current round's ``x_hat``.

For image metrics, render each round's ``x_hat`` into an ``images`` subdirectory:

    uv run --extra render python experiments/generate_images.py \
        outputs/cycles/validation/round_00/x_and_x_hat.pt \
        --output-dir outputs/cycles/validation/round_00/images \
        --rank 1 --fields x_hat --no-scores

Then evaluate the complete trajectory:

    uv run --extra evaluation python experiments/cycle/evaluate_cycles.py \
        outputs/cycles/validation --rows 0 1 2 3

Pass ``--no-images`` for weight-only evaluation. All ``.pt`` inputs are
pickle-backed and must come from trusted sources.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


LONG_FIELDS = [
    "round",
    "payload_row",
    "sample_id",
    "w_mse_step",
    "w_r2_step",
    "w_cos_step",
    "w_mse_cum",
    "w_r2_cum",
    "w_cos_cum",
    "w_relnorm_cum",
    "img_mse_step",
    "ssim_step",
    "lpips_step",
    "img_mse_cum",
    "ssim_cum",
    "lpips_cum",
]
METRIC_FIELDS = [
    field
    for field in LONG_FIELDS
    if field not in {"round", "payload_row", "sample_id"}
]
ROUND_PATTERN = re.compile(r"round_(\d+)$")


@dataclass(frozen=True)
class RoundArtifact:
    number: int
    name: str
    x: torch.Tensor
    x_hat: torch.Tensor
    indices: list[Any]
    image_dir: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cycle_root",
        type=Path,
        help="Directory containing round_XX cycle artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Evaluation output directory (default: CYCLE_ROOT/evaluation).",
    )
    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        help="Consecutive rounds beginning at zero (default: discover all).",
    )
    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        help="Zero-based payload rows to evaluate (default: all).",
    )
    parser.add_argument(
        "--images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute image-space metrics from rendered PNGs (default: true).",
    )
    parser.add_argument(
        "--lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include AlexNet LPIPS when image metrics are enabled (default: true).",
    )
    parser.add_argument(
        "--figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write PDF and PNG summary figures (default: true).",
    )
    parser.add_argument(
        "--image-dir",
        default="images",
        help="Image subdirectory within every round directory (default: images).",
    )
    parser.add_argument(
        "--image-template",
        default="sample_{sid}_x_hat.png",
        help="Current-round image filename template; supports {sid}.",
    )
    parser.add_argument(
        "--reference-image-dir",
        type=Path,
        help="Optional external directory containing original x renders.",
    )
    parser.add_argument(
        "--reference-image-template",
        default="sample_{sid}_x.png",
        help="External reference filename template; supports {sid}.",
    )
    parser.add_argument(
        "--grid-rounds",
        nargs="+",
        type=int,
        help="Rounds shown in the qualitative grid (default: all evaluated rounds).",
    )
    parser.add_argument("--run-label", default="Encoding-decoding cycles")
    parser.add_argument(
        "--device",
        default="auto",
        help="LPIPS device, e.g. cuda:0 or cpu (default: auto).",
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested, but CUDA is unavailable")
    return device


def _python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return value


def _safe_identifier(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _normalise_indices(values: Any, sample_count: int) -> list[Any]:
    if values is None:
        return list(range(sample_count))
    if len(values) != sample_count:
        raise ValueError(
            f"indices contains {len(values)} entries for {sample_count} samples"
        )
    return [_python_scalar(value) for value in values]


def discover_round_numbers(root: Path) -> list[int]:
    numbers = []
    if not root.is_dir():
        raise FileNotFoundError(f"Cycle root not found: {root}")
    for child in root.iterdir():
        match = ROUND_PATTERN.fullmatch(child.name)
        if child.is_dir() and match and (child / "x_and_x_hat.pt").is_file():
            numbers.append(int(match.group(1)))
    if not numbers:
        raise FileNotFoundError(f"No round_XX directories found under {root}")
    return sorted(numbers)


def validate_round_numbers(numbers: Sequence[int]) -> list[int]:
    selected = list(numbers)
    if len(set(selected)) != len(selected):
        raise ValueError("Round selection contains duplicates")
    selected.sort()
    expected = list(range(selected[-1] + 1)) if selected else []
    if selected != expected:
        raise ValueError(
            "Cycle evaluation requires consecutive rounds beginning at 0; "
            f"got {selected}"
        )
    return selected


def load_rounds(args: argparse.Namespace) -> list[RoundArtifact]:
    numbers = validate_round_numbers(
        args.rounds if args.rounds is not None else discover_round_numbers(args.cycle_root)
    )
    rounds = []
    for number in numbers:
        name = f"round_{number:02d}"
        path = args.cycle_root / name / "x_and_x_hat.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing cycle payload: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Cycle payload must be a dictionary: {path}")

        x = payload.get("x")
        x_hat = payload.get("x_hat")
        if not isinstance(x, torch.Tensor) or not isinstance(x_hat, torch.Tensor):
            raise ValueError(f"Cycle payload must contain tensor x and x_hat: {path}")
        x = x.reshape(x.shape[0], -1).float()
        x_hat = x_hat.reshape(x_hat.shape[0], -1).float()
        if x.shape != x_hat.shape:
            raise ValueError(f"{path}: x {tuple(x.shape)} != x_hat {tuple(x_hat.shape)}")
        stored_round = payload.get("round", number)
        if int(stored_round) != number:
            raise ValueError(
                f"{path}: payload round is {stored_round}, directory round is {number}"
            )

        rounds.append(
            RoundArtifact(
                number=number,
                name=name,
                x=x,
                x_hat=x_hat,
                indices=_normalise_indices(payload.get("indices"), x.shape[0]),
                image_dir=args.cycle_root / name / args.image_dir,
            )
        )
        print(f"Loaded {name}: {tuple(x.shape)}")
    return rounds


def validate_artifact_chain(rounds: Sequence[RoundArtifact]) -> None:
    anchor = rounds[0]
    if not torch.allclose(anchor.x, anchor.x_hat, atol=1e-7, rtol=0):
        deviation = (anchor.x - anchor.x_hat).abs().max().item()
        raise ValueError(
            f"Round zero is not an identity baseline (max difference {deviation:.3e})"
        )

    for artifact in rounds[1:]:
        if artifact.x.shape != anchor.x.shape:
            raise ValueError(
                f"{artifact.name} shape {tuple(artifact.x.shape)} differs from "
                f"round zero {tuple(anchor.x.shape)}"
            )
        if artifact.indices != anchor.indices:
            raise ValueError(f"{artifact.name} sample indices differ from round zero")
        if not torch.allclose(artifact.x, anchor.x, atol=1e-7, rtol=0):
            deviation = (artifact.x - anchor.x).abs().max().item()
            raise ValueError(
                f"{artifact.name} does not preserve x_0 in payload['x'] "
                f"(max difference {deviation:.3e})"
            )
    print("Validated cycle artifacts: every round shares the same x_0 anchor.")


def select_rows(rows: Sequence[int] | None, sample_count: int) -> list[int]:
    selected = list(range(sample_count)) if rows is None else list(rows)
    invalid = [row for row in selected if row < 0 or row >= sample_count]
    if invalid:
        raise IndexError(
            f"Payload rows {invalid} are outside the 0..{sample_count - 1} range"
        )
    if not selected:
        raise ValueError("No payload rows were selected")
    if len(set(selected)) != len(selected):
        raise ValueError("--rows contains duplicate row indices")
    return selected


def weight_metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    mse = torch.mean((a - b) ** 2).item()
    ss_res = torch.sum((a - b) ** 2).item()
    ss_tot = torch.sum((a - a.mean()) ** 2).item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    denominator = (a.norm() * b.norm()).item()
    cosine = torch.dot(a, b).item() / denominator if denominator > 0 else float("nan")
    return mse, r2, cosine


def relative_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    norm = a.norm().item()
    return (b - a).norm().item() / norm if norm > 0 else float("nan")


def load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required for image evaluation. Install it with "
            "`uv sync --extra evaluation`."
        ) from error
    if not path.is_file():
        raise FileNotFoundError(f"Missing rendered cycle image: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def image_to_lpips_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(array.astype(np.float32) / 127.5 - 1.0)
    return tensor.permute(2, 0, 1).unsqueeze(0).to(device)


def image_metrics(
    first: np.ndarray,
    second: np.ndarray,
    lpips_model: Any,
    device: torch.device,
) -> tuple[float, float, float]:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:
        raise RuntimeError(
            "scikit-image is required for image evaluation. Install it with "
            "`uv sync --extra evaluation`."
        ) from error
    if first.shape != second.shape:
        raise ValueError(f"Image shape mismatch: {first.shape} != {second.shape}")

    first_float = first.astype(np.float32)
    second_float = second.astype(np.float32)
    mse = float(np.mean((first_float - second_float) ** 2))
    ssim = float(
        structural_similarity(
            first_float,
            second_float,
            channel_axis=2,
            data_range=255.0,
        )
    )
    lpips = float("nan")
    if lpips_model is not None:
        with torch.inference_mode():
            lpips = float(
                lpips_model(
                    image_to_lpips_tensor(first, device),
                    image_to_lpips_tensor(second, device),
                ).squeeze().item()
            )
    return mse, ssim, lpips


def load_lpips_model(device: torch.device) -> Any:
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError(
            "LPIPS is required unless --no-lpips is passed. Install it with "
            "`uv sync --extra evaluation`."
        ) from error
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


def evaluate(
    rounds: Sequence[RoundArtifact],
    rows: Sequence[int],
    args: argparse.Namespace,
    lpips_model: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    anchor = rounds[0]
    image_cache: dict[tuple[int, int], np.ndarray] = {}
    reference_cache: dict[int, np.ndarray] = {}

    def current_image(artifact: RoundArtifact, row: int) -> np.ndarray:
        key = (artifact.number, row)
        if key not in image_cache:
            sample_id = _safe_identifier(anchor.indices[row])
            image_cache[key] = load_image(
                artifact.image_dir / args.image_template.format(sid=sample_id)
            )
        return image_cache[key]

    def reference_image(row: int) -> np.ndarray:
        if row not in reference_cache:
            if args.reference_image_dir is None:
                reference_cache[row] = current_image(anchor, row)
            else:
                sample_id = _safe_identifier(anchor.indices[row])
                reference_cache[row] = load_image(
                    args.reference_image_dir
                    / args.reference_image_template.format(sid=sample_id)
                )
        return reference_cache[row]

    results = []
    for round_index, artifact in enumerate(rounds):
        previous = rounds[round_index - 1] if round_index > 0 else None
        print(f"Evaluating {artifact.name}")
        for row in rows:
            sample_id = anchor.indices[row]
            current_weights = artifact.x_hat[row]
            cumulative = weight_metrics(anchor.x[row], current_weights)
            cumulative_relative_norm = relative_norm(anchor.x[row], current_weights)
            if previous is None:
                step = (float("nan"), float("nan"), float("nan"))
            else:
                step = weight_metrics(previous.x_hat[row], current_weights)

            image_step = (float("nan"), float("nan"), float("nan"))
            image_cumulative = (float("nan"), float("nan"), float("nan"))
            if args.images:
                current = current_image(artifact, row)
                image_cumulative = image_metrics(
                    reference_image(row), current, lpips_model, device
                )
                if previous is not None:
                    image_step = image_metrics(
                        current_image(previous, row), current, lpips_model, device
                    )

            result = {
                "round": artifact.number,
                "payload_row": row,
                "sample_id": sample_id,
                "w_mse_step": step[0],
                "w_r2_step": step[1],
                "w_cos_step": step[2],
                "w_mse_cum": cumulative[0],
                "w_r2_cum": cumulative[1],
                "w_cos_cum": cumulative[2],
                "w_relnorm_cum": cumulative_relative_norm,
                "img_mse_step": image_step[0],
                "ssim_step": image_step[1],
                "lpips_step": image_step[2],
                "img_mse_cum": image_cumulative[0],
                "ssim_cum": image_cumulative[1],
                "lpips_cum": image_cumulative[2],
            }
            results.append(result)
            print(
                f"  row={row}, id={sample_id}: cumulative weight R2="
                f"{cumulative[1]:.4f}, step weight R2={step[1]:.4f}"
            )
    return results


def metric_statistics(
    results: Sequence[dict[str, Any]],
) -> dict[int, dict[str, tuple[float, float]]]:
    statistics = {}
    for round_number in sorted({int(row["round"]) for row in results}):
        subset = [row for row in results if row["round"] == round_number]
        round_stats = {}
        for metric in METRIC_FIELDS:
            values = np.asarray([row[metric] for row in subset], dtype=float)
            values = values[~np.isnan(values)]
            round_stats[metric] = (
                (float(values.mean()), float(values.std(ddof=0)))
                if values.size
                else (float("nan"), float("nan"))
            )
        statistics[round_number] = round_stats
    return statistics


def _format(value: float, decimals: int = 4) -> str:
    return "" if math.isnan(value) else f"{value:.{decimals}f}"


def write_csvs(
    results: Sequence[dict[str, Any]],
    output_dir: Path,
    sample_count: int,
) -> dict[int, dict[str, tuple[float, float]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "cycle_scores_long.csv"
    with long_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LONG_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    key: round(value, 8) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    statistics = metric_statistics(results)
    round_path = output_dir / "cycle_scores_by_round.csv"
    with round_path.open("w", newline="", encoding="utf-8") as csv_file:
        header = ["round", "n_samples"]
        for metric in METRIC_FIELDS:
            header.extend((f"{metric}_mean", f"{metric}_std"))
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for round_number, round_stats in statistics.items():
            line: list[Any] = [round_number, sample_count]
            for metric in METRIC_FIELDS:
                mean, std = round_stats[metric]
                line.extend((round(mean, 8), round(std, 8)))
            writer.writerow(line)

    table_path = output_dir / "cycle_table_paper.csv"
    table_columns = [
        ("w_r2_cum", "Weight R2 vs. original", 4),
        ("w_mse_cum", "Weight MSE vs. original", 6),
        ("w_relnorm_cum", "Relative weight drift", 4),
        ("ssim_cum", "SSIM vs. original", 4),
        ("lpips_cum", "LPIPS vs. original", 4),
        ("img_mse_cum", "Pixel MSE vs. original", 1),
        ("ssim_step", "SSIM vs. previous round", 4),
        ("lpips_step", "LPIPS vs. previous round", 4),
    ]
    with table_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Round"] + [label for _, label, _ in table_columns])
        for round_number, round_stats in statistics.items():
            line: list[Any] = [round_number]
            for metric, _, decimals in table_columns:
                mean, std = round_stats[metric]
                line.append(
                    ""
                    if math.isnan(mean)
                    else f"{_format(mean, decimals)} ± {_format(std, decimals)}"
                )
            writer.writerow(line)

    print(f"Wrote {long_path}")
    print(f"Wrote {round_path}")
    print(f"Wrote {table_path}")
    return statistics


def setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.4,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )
    return plt


def write_metric_figure(
    results: Sequence[dict[str, Any]],
    rows: Sequence[int],
    output_dir: Path,
    run_label: str,
    include_images: bool,
    include_lpips: bool,
) -> None:
    plt = setup_matplotlib()
    round_numbers = sorted({int(result["round"]) for result in results})
    panels = [("w_r2_cum", r"Weight-space $R^2$ vs. original", r"$R^2$")]
    if include_images:
        panels.append(("ssim_cum", "SSIM vs. original image", "SSIM"))
        if include_lpips:
            panels.append(("lpips_cum", "LPIPS vs. original image", "LPIPS"))
        panels.append(("ssim_step", "SSIM vs. previous round", "SSIM"))

    column_count = 2 if len(panels) > 1 else 1
    row_count = math.ceil(len(panels) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6.3 if column_count == 2 else 3.2, 2.2 * row_count),
        sharex=True,
        squeeze=False,
    )
    flat_axes = list(axes.ravel())
    x_values = np.asarray(round_numbers)
    for axis, (metric, title, y_label) in zip(flat_axes, panels):
        per_row = {}
        for payload_row in rows:
            per_row[payload_row] = [
                next(
                    result[metric]
                    for result in results
                    if result["round"] == round_number
                    and result["payload_row"] == payload_row
                )
                for round_number in round_numbers
            ]
            axis.plot(x_values, per_row[payload_row], color="0.72", linewidth=0.7)
        values = np.asarray(list(per_row.values()), dtype=float)
        means = np.asarray(
            [
                np.nan if np.isnan(column).all() else np.nanmean(column)
                for column in values.T
            ]
        )
        stds = np.asarray(
            [
                np.nan if np.isnan(column).all() else np.nanstd(column)
                for column in values.T
            ]
        )
        axis.fill_between(x_values, means - stds, means + stds, alpha=0.15)
        axis.plot(x_values, means, marker="o", markersize=3.2, label="mean")
        axis.set_title(title)
        axis.set_ylabel(y_label)
        axis.set_xticks(x_values)
        axis.set_xlabel("Cycle round")
    for axis in flat_axes[len(panels) :]:
        axis.remove()
    flat_axes[0].legend(frameon=False, loc="best")
    figure.suptitle(run_label, y=1.01, fontsize=10)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        path = output_dir / f"fig_cycle_metrics.{extension}"
        figure.savefig(path, dpi=300)
        print(f"Wrote {path}")
    plt.close(figure)


def write_image_grid(
    rounds: Sequence[RoundArtifact],
    rows: Sequence[int],
    grid_rounds: Sequence[int] | None,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    selected_numbers = (
        [artifact.number for artifact in rounds]
        if grid_rounds is None
        else list(grid_rounds)
    )
    artifacts = {artifact.number: artifact for artifact in rounds}
    missing = [number for number in selected_numbers if number not in artifacts]
    if missing:
        raise ValueError(f"--grid-rounds includes unevaluated rounds: {missing}")

    plt = setup_matplotlib()
    figure, axes = plt.subplots(
        len(rows),
        len(selected_numbers),
        figsize=(min(1.05 * len(selected_numbers), 9.0), 1.05 * len(rows) + 0.35),
        gridspec_kw={"wspace": 0.04, "hspace": 0.04},
        squeeze=False,
    )
    anchor = rounds[0]
    for row_index, payload_row in enumerate(rows):
        sample_id = anchor.indices[payload_row]
        safe_id = _safe_identifier(sample_id)
        for column_index, round_number in enumerate(selected_numbers):
            axis = axes[row_index, column_index]
            path = (
                artifacts[round_number].image_dir
                / args.image_template.format(sid=safe_id)
            )
            axis.imshow(load_image(path))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("0.7")
            if row_index == 0:
                axis.set_title(f"r{round_number}", pad=2.5, fontsize=8)
            if column_index == 0:
                axis.set_ylabel(f"id {sample_id}", fontsize=8, labelpad=3)

    figure.suptitle(
        f"{args.run_label}: identity drift across cycle rounds",
        y=0.995,
        fontsize=10,
    )
    for extension in ("pdf", "png"):
        path = output_dir / f"fig_cycle_grid.{extension}"
        figure.savefig(path, dpi=300)
        print(f"Wrote {path}")
    plt.close(figure)


def write_evaluation_config(
    args: argparse.Namespace,
    output_dir: Path,
    rounds: Sequence[RoundArtifact],
    rows: Sequence[int],
    device: torch.device,
) -> None:
    config = {
        "cycle_root": str(args.cycle_root.resolve()),
        "rounds": [artifact.number for artifact in rounds],
        "rows": list(rows),
        "sample_ids": [rounds[0].indices[row] for row in rows],
        "images": args.images,
        "lpips": args.lpips,
        "figures": args.figures,
        "image_dir": args.image_dir,
        "image_template": args.image_template,
        "reference_image_dir": (
            str(args.reference_image_dir.resolve())
            if args.reference_image_dir is not None
            else None
        ),
        "reference_image_template": args.reference_image_template,
        "grid_rounds": args.grid_rounds,
        "run_label": args.run_label,
        "device": str(device),
    }
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    if args.lpips and not args.images:
        raise ValueError("--lpips requires image evaluation; use --no-lpips --no-images")
    rounds = load_rounds(args)
    validate_artifact_chain(rounds)
    rows = select_rows(args.rows, rounds[0].x.shape[0])
    output_dir = args.output_dir or args.cycle_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    lpips_model = load_lpips_model(device) if args.images and args.lpips else None

    write_evaluation_config(args, output_dir, rounds, rows, device)
    results = evaluate(rounds, rows, args, lpips_model, device)
    write_csvs(results, output_dir, len(rows))
    if args.figures:
        write_metric_figure(
            results,
            rows,
            output_dir,
            args.run_label,
            args.images,
            args.lpips,
        )
        if args.images:
            write_image_grid(rounds, rows, args.grid_rounds, args, output_dir)
    print(f"Done. Evaluation written to {output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

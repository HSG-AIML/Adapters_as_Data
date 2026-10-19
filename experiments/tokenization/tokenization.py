"""Shared loading and print style for the tokenization figures."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


PRINT_STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 150,
}


def apply_print_style(usetex: bool = False) -> None:
    """Match the LaTeX document's typography."""

    import matplotlib as mpl

    mpl.rcParams.update({**PRINT_STYLE, "text.usetex": usetex})


def load_adapter_vector(path: Path, field: str, row: int) -> torch.Tensor:
    """Return one adapter of a flat weight population as a ``[D]`` vector."""

    if not path.is_file():
        raise FileNotFoundError(f"Weight payload not found: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, torch.Tensor):
        weights = loaded
    elif isinstance(loaded, Mapping):
        if field not in loaded:
            raise KeyError(
                f"Input payload has no {field!r} field; available fields: "
                f"{sorted(str(key) for key in loaded)}"
            )
        weights = loaded[field]
    else:
        raise ValueError("Weight input must be a tensor or dictionary")

    if not isinstance(weights, torch.Tensor) or weights.ndim < 2:
        raise ValueError(f"Weight field {field!r} must have a sample axis")
    weights = weights.reshape(weights.shape[0], -1)
    if row < 0 or row >= weights.shape[0]:
        raise IndexError(f"--row {row} is outside 0..{weights.shape[0] - 1}")
    return weights[row].detach().float()


def robust_limits(grid: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Colour scale over the real (unpadded) weights only."""

    low, high = np.percentile(grid[mask], [1, 99])
    return float(low), float(high)


def save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    print(f"Saved figure to {path}")

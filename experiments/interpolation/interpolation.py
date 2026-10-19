"""Shared linear interpolation for matched latent- and weight-space experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


DEFAULT_STEPS = 10


def interpolation_alphas(steps: int) -> list[float]:
    """Evenly spaced blend factors from 0 to 1, including both endpoints."""

    if steps < 2:
        raise ValueError("--steps must be at least 2 to include both endpoints")
    return [index / (steps - 1) for index in range(steps)]


def interpolate(
    sample_a: torch.Tensor,
    sample_b: torch.Tensor,
    alphas: Sequence[float],
) -> torch.Tensor:
    """Stack ``(1 - alpha) * A + alpha * B`` for every alpha along a new axis."""

    if sample_a.shape != sample_b.shape:
        raise ValueError(
            f"Interpolation inputs must have equal shapes: "
            f"{tuple(sample_a.shape)} != {tuple(sample_b.shape)}"
        )
    return torch.stack(
        [(1.0 - alpha) * sample_a + alpha * sample_b for alpha in alphas],
        dim=0,
    )


def pair_geometry(
    sample_a: torch.Tensor,
    sample_b: torch.Tensor,
) -> tuple[float, float]:
    """L2 distance and cosine similarity between the two anchor samples."""

    l2_distance = (sample_a - sample_b).norm().item()
    cosine = torch.nn.functional.cosine_similarity(
        sample_a.reshape(1, -1),
        sample_b.reshape(1, -1),
    ).item()
    return l2_distance, cosine


def tensor_stats(tensor: torch.Tensor) -> str:
    values = tensor.float()
    return (
        f"mean={values.mean():.6f}  std={values.std():.6f}  "
        f"min={values.min():.6f}  max={values.max():.6f}"
    )


def python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return value


def sample_ids(payload: dict[str, Any], sample_count: int) -> list[Any]:
    values = payload.get("ids", payload.get("indices"))
    if values is None:
        return list(range(sample_count))
    if len(values) != sample_count:
        raise ValueError(
            f"Payload IDs contain {len(values)} entries for {sample_count} samples"
        )
    return [python_scalar(value) for value in values]


def validate_sample_pair(sample_a: int, sample_b: int, sample_count: int) -> None:
    invalid = [index for index in (sample_a, sample_b) if index < 0 or index >= sample_count]
    if invalid:
        raise IndexError(
            f"Sample rows {invalid} are outside the 0..{sample_count - 1} range"
        )
    if sample_a == sample_b:
        raise ValueError("Sample A and sample B must be different rows")


def tile_rows(tensor: torch.Tensor, count: int) -> torch.Tensor:
    """Repeat a single sample row ``count`` times along a new leading axis."""

    return tensor.unsqueeze(0).repeat(count, *([1] * tensor.dim()))

"""Shared tensor arithmetic for matched latent- and weight-space experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


OPERATION_NAMES = (
    "diff_a_minus_b",
    "diff_b_minus_a",
    "sum",
    "mean",
    "add",
)
DEFAULT_OPERATIONS = OPERATION_NAMES
DEFAULT_ADD_SCALES = (0.5, 1.5)


def apply_operation(
    sample_a: torch.Tensor,
    sample_b: torch.Tensor,
    operation: str,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, str]:
    """Apply one of the paper's pairwise arithmetic operations."""

    if sample_a.shape != sample_b.shape:
        raise ValueError(
            f"Arithmetic inputs must have equal shapes: "
            f"{tuple(sample_a.shape)} != {tuple(sample_b.shape)}"
        )
    if operation == "diff_a_minus_b":
        return sample_a - sample_b, "A - B"
    if operation == "diff_b_minus_a":
        return sample_b - sample_a, "B - A"
    if operation == "sum":
        return sample_a + sample_b, "A + B"
    if operation == "mean":
        return (sample_a + sample_b) / 2.0, "(A + B) / 2"
    if operation == "add":
        if scale is None:
            raise ValueError("The add operation requires a scale")
        return sample_a + scale * sample_b, f"A + {scale:g} * B"
    raise ValueError(
        f"Unknown operation {operation!r}; choose from {', '.join(OPERATION_NAMES)}"
    )


def operation_specs(
    operations: Sequence[str],
    add_scales: Sequence[float],
) -> list[tuple[str, dict[str, float]]]:
    """Expand the selected operations into concrete operation/argument pairs."""

    selected = list(dict.fromkeys(operations))
    unknown = [operation for operation in selected if operation not in OPERATION_NAMES]
    if unknown:
        raise ValueError(f"Unknown arithmetic operations: {unknown}")

    scales = list(add_scales)
    if "add" in selected and not scales:
        raise ValueError("At least one --add-scales value is required for add")
    if len(set(scales)) != len(scales):
        raise ValueError("--add-scales contains duplicate values")

    specs = []
    for operation in selected:
        if operation == "add":
            specs.extend((operation, {"scale": scale}) for scale in scales)
        else:
            specs.append((operation, {}))
    return specs


def operation_filename(
    operation: str,
    kwargs: dict[str, float],
    sample_a: int,
    sample_b: int,
) -> str:
    arguments = "_".join(f"{key}{value:g}" for key, value in kwargs.items())
    operation_part = f"{operation}_{arguments}" if arguments else operation
    return f"{operation_part}_A{sample_a}_B{sample_b}.pt"


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

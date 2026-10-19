"""Shared Gaussian perturbation for matched latent- and weight-space experiments."""

from __future__ import annotations

from typing import Any

import torch


DEFAULT_SEED = 42


def summarize(tensor: torch.Tensor) -> dict[str, Any]:
    """Global and per-coordinate summary statistics of a population tensor."""

    values = tensor.float()
    leading = tuple(range(values.ndim - 1))
    coordinate_std = values.std(dim=leading)
    return {
        "shape": tuple(values.shape),
        "mean": values.mean().item(),
        "std": values.std().item(),
        "variance": values.var().item(),
        "min": values.min().item(),
        "max": values.max().item(),
        "median": values.median().item(),
        "dim_mean_mean": values.mean(dim=leading).mean().item(),
        "dim_std_mean": coordinate_std.mean().item(),
        "dim_std_min": coordinate_std.min().item(),
        "dim_std_max": coordinate_std.max().item(),
    }


def print_statistics(stats: dict[str, Any], tag: str) -> None:
    print(f"  [{tag}] statistics")
    print(f"    shape           : {stats['shape']}")
    print(f"    mean            : {stats['mean']:.6f}")
    print(f"    std             : {stats['std']:.6f}")
    print(f"    variance        : {stats['variance']:.6f}")
    print(f"    min / max       : {stats['min']:.6f} / {stats['max']:.6f}")
    print(f"    median          : {stats['median']:.6f}")
    print(
        f"    per-dim std     : mean={stats['dim_std_mean']:.6f}  "
        f"min={stats['dim_std_min']:.6f}  max={stats['dim_std_max']:.6f}"
    )


def add_gaussian_noise(
    tensor: torch.Tensor,
    *,
    scale: float,
    sigma: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    """Add ``scale * sigma * N(0, 1)`` noise, returning the perturbed tensor.

    ``sigma`` is the population spread the scale is expressed in, so the axis is
    the same in both spaces: a scale of 1 perturbs by one population standard
    deviation.
    """

    values = tensor.float()
    noise_std = scale * sigma
    noise = torch.randn(
        values.shape,
        generator=generator,
        dtype=values.dtype,
    )
    return (values + noise_std * noise).to(tensor.dtype), noise_std

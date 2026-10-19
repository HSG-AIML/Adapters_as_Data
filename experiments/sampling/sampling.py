"""Shared Gaussian and PCA sampling operations for latent and weight spaces."""

from __future__ import annotations

import torch


def fit_pca(
    flattened: torch.Tensor,
    n_components: int,
) -> dict[str, torch.Tensor]:
    """Fit centered PCA with an economy SVD."""

    if flattened.ndim != 2 or flattened.shape[0] < 2:
        raise ValueError("PCA input must have shape [N, D] with N >= 2")
    if n_components < 1 or n_components > min(flattened.shape):
        raise ValueError(
            f"n_components must be in 1..{min(flattened.shape)}, got {n_components}"
        )

    mean = flattened.mean(dim=0)
    centered = flattened - mean
    _, singular_values, right_vectors = torch.linalg.svd(
        centered,
        full_matrices=False,
    )
    components = right_vectors[:n_components].T
    projection = centered @ components
    total_variance = singular_values.square().sum()
    explained_variance_ratio = (
        singular_values[:n_components].square() / total_variance
    )
    return {
        "mean": mean,
        "components": components,
        "projection": projection,
        "explained_variance_ratio": explained_variance_ratio,
    }


def sample_naive(
    flattened: torch.Tensor,
    *,
    num_samples: int,
    factor: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample independent empirical Gaussians for every input coordinate."""

    coordinate_mean = flattened.mean(dim=0)
    coordinate_std = flattened.std(dim=0)
    noise = torch.randn(
        num_samples,
        flattened.shape[1],
        generator=generator,
    )
    samples = coordinate_mean.unsqueeze(0) + (
        factor * coordinate_std.unsqueeze(0) * noise
    )
    return samples, {
        "coordinate_mean": coordinate_mean,
        "coordinate_std": coordinate_std,
    }


def sample_pca(
    fitted: dict[str, torch.Tensor],
    *,
    num_samples: int,
    factor: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample empirical Gaussian coefficients in a fitted PCA basis."""

    projection = fitted["projection"]
    coefficient_mean = projection.mean(dim=0)
    coefficient_std = projection.std(dim=0)
    noise = torch.randn(
        num_samples,
        projection.shape[1],
        generator=generator,
    )
    coefficients = coefficient_mean.unsqueeze(0) + (
        factor * coefficient_std.unsqueeze(0) * noise
    )
    samples = coefficients @ fitted["components"].T + fitted["mean"].unsqueeze(0)
    parameters = dict(fitted)
    parameters.update(
        {
            "coefficient_mean": coefficient_mean,
            "coefficient_std": coefficient_std,
        }
    )
    return samples, parameters


def explained_variance_summary(ratios: torch.Tensor) -> dict[str, float]:
    """Return cumulative explained variance at standard reporting cutoffs."""

    cumulative = ratios.cumsum(dim=0)
    checkpoints = sorted({1, 5, 10, 20, 50, 256, ratios.numel()})
    return {
        f"top_{count}": float(cumulative[count - 1].item())
        for count in checkpoints
        if count <= ratios.numel()
    }

"""Plot the dense tokenization of a single flat LoRA adapter.

The adapter is chunked into fixed-width tokens as a contiguous stream, so only
the final token is padded; that tail is the only structure marked.

Example:

    uv run python experiments/tokenization/plot_dense_tokenization.py \
        outputs/reconstructions/x_and_x_hat_val.pt \
        --field x --row 4 \
        --output figures/dense_tokenization.pdf

Torch payloads are pickle-backed and must come from trusted sources.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from tokenization import (
    apply_print_style,
    load_adapter_vector,
    robust_limits,
    save_figure,
)


DEFAULT_TOKEN_SIZE = 320  # smallest channel width; the dense token width


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
    parser.add_argument("--row", type=int, default=4, help="Adapter row to plot.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/dense_tokenization.pdf"),
    )
    parser.add_argument(
        "--token-size",
        type=int,
        default=DEFAULT_TOKEN_SIZE,
        help=f"Dense token width (default: {DEFAULT_TOKEN_SIZE}).",
    )
    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Render text with a local LaTeX toolchain.",
    )
    return parser.parse_args(argv)


def tokenize_dense(
    vector: torch.Tensor,
    token_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Chunk the stream into fixed-size tokens, padding only the global tail."""

    width = vector.numel()
    token_count = int(np.ceil(width / token_size))
    padding = token_count * token_size - width
    padded = torch.cat([vector, torch.zeros(padding)])
    tokens = padded.view(token_count, token_size).numpy()
    mask = np.ones((token_count, token_size), dtype=bool)
    if padding > 0:
        mask[-1, token_size - padding:] = False
    return tokens, mask


def plot_dense(
    vector: torch.Tensor,
    token_size: int,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    tokens, mask = tokenize_dense(vector, token_size)
    token_count = tokens.shape[0]
    grid = tokens.T  # x = token index, y = value within token
    vmin, vmax = robust_limits(grid, mask.T)

    figure, axes = plt.subplots(figsize=(6.3, 2.4), constrained_layout=True)
    image = axes.imshow(
        grid,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
        origin="lower",
    )

    pad_cells = int((~mask[-1]).sum())
    if pad_cells > 0:
        first_pad = token_size - pad_cells
        axes.add_patch(
            Rectangle(
                (token_count - 1 - 0.5, first_pad - 0.5),
                1,
                pad_cells,
                fill=False,
                hatch="////",
                edgecolor="grey",
                linewidth=0.6,
            )
        )
        axes.annotate(
            "padding",
            xy=(token_count - 1, first_pad + pad_cells / 2),
            xytext=(token_count - 1 - 30, first_pad + pad_cells / 2),
            fontsize=10,
            color="white",
            va="center",
            ha="right",
            arrowprops=dict(arrowstyle="->", color="white", lw=0.6),
        )

    axes.set_xlabel("token index")
    axes.set_ylabel("value within token")
    axes.set_title(f"Dense tokenization — {token_count} tokens of size {token_size}")
    figure.colorbar(
        image,
        ax=axes,
        label="weight value",
        shrink=0.9,
        location="right",
        pad=0.02,
    )

    save_figure(figure, output)
    plt.close(figure)
    print(
        f"  {token_count} tokens x {token_size} dims; "
        f"last token padded by {pad_cells} cells"
    )


def run(args: argparse.Namespace) -> None:
    if args.token_size < 1:
        raise ValueError("--token-size must be positive")
    apply_print_style(args.usetex)
    vector = load_adapter_vector(args.weights, args.field, args.row)
    print(f"Loaded {args.weights} row {args.row}: {vector.numel():,} weights")
    plot_dense(vector, args.token_size, args.output)


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

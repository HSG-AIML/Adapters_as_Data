"""Plot the sparse tokenization of a single flat LoRA adapter.

Unlike dense tokenization, the sparse scheme respects matrix boundaries: each
weight matrix is sliced independently and only its last token is padded, so most
tokens carry padding. Padded cells are drawn in grey and matrix boundaries are
marked.

Example:

    uv run python experiments/tokenization/plot_sparse_tokenization.py \
        outputs/reconstructions/x_and_x_hat_val.pt \
        --weight-dimensions path/to/weight_dimensions.pt \
        --field x --row 4 \
        --output figures/sparse_tokenization.pdf

Torch payloads are pickle-backed and must come from trusted sources.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from sane.data.tokenizers import SparseVectorTokenizer

from tokenization import (
    apply_print_style,
    load_adapter_vector,
    robust_limits,
    save_figure,
)


DEFAULT_TOKEN_SIZE = 768  # largest matrix dim (cross-attention text A)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "weights",
        type=Path,
        help="Trusted tensor or payload containing a flat adapter population.",
    )
    parser.add_argument(
        "--weight-dimensions",
        type=Path,
        required=True,
        help="Trusted weight_dimensions.pt defining the matrix boundaries.",
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
        default=Path("figures/sparse_tokenization.pdf"),
    )
    parser.add_argument(
        "--token-size",
        type=int,
        default=DEFAULT_TOKEN_SIZE,
        help=f"Sparse token width (default: {DEFAULT_TOKEN_SIZE}).",
    )
    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Render text with a local LaTeX toolchain.",
    )
    return parser.parse_args(argv)


def tokenize_sparse(
    vector: torch.Tensor,
    tokenizer: SparseVectorTokenizer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice per matrix with the real tokenizer.

    Returns the tokens, their mask, and the token indices at which a new matrix
    begins.
    """

    sliced = tokenizer.slice_by_layers(tokenizer.flatten({"vector": vector}))
    token_chunks, mask_chunks, matrices = [], [], []
    for matrix_index, (tokens, mask, _positions) in enumerate(sliced):
        token_chunks.append(tokens)
        mask_chunks.append(mask)
        matrices += [matrix_index] * tokens.shape[0]

    matrices = np.array(matrices)
    boundaries = np.where(np.diff(matrices, prepend=matrices[0] - 1) != 0)[0]
    return (
        torch.cat(token_chunks, dim=0).numpy(),
        torch.cat(mask_chunks, dim=0).numpy(),
        boundaries,
    )


def plot_sparse(
    vector: torch.Tensor,
    tokenizer: SparseVectorTokenizer,
    output: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    tokens, mask, boundaries = tokenize_sparse(vector, tokenizer)
    token_count, token_size = tokens.shape
    grid = tokens.T  # x = token index, y = value within token
    grid_mask = mask.T
    vmin, vmax = robust_limits(grid, grid_mask)

    colormap = mpl.cm.viridis.copy()
    colormap.set_bad(color="0.85")  # padded cells shown in light grey

    figure, axes = plt.subplots(figsize=(6.3, 2.6), constrained_layout=True)
    image = axes.imshow(
        np.ma.array(grid, mask=~grid_mask),
        aspect="auto",
        cmap=colormap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
        origin="lower",
    )
    for boundary in boundaries[1:]:  # skip the first, at column 0
        axes.axvline(boundary - 0.5, color="white", linewidth=0.3, alpha=0.7)

    axes.set_xlabel("token index")
    axes.set_ylabel("value within token")
    axes.set_title(f"Sparse tokenization — {token_count} tokens of size {token_size}")
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
    padded_tokens = int((~mask.all(axis=1)).sum())
    print(
        f"  {token_count} tokens x {token_size} dims; "
        f"{padded_tokens}/{token_count} tokens carry padding"
    )


def run(args: argparse.Namespace) -> None:
    if args.token_size < 1:
        raise ValueError("--token-size must be positive")
    if not args.weight_dimensions.is_file():
        raise FileNotFoundError(
            f"Weight dimensions not found: {args.weight_dimensions}"
        )
    apply_print_style(args.usetex)
    vector = load_adapter_vector(args.weights, args.field, args.row)
    print(f"Loaded {args.weights} row {args.row}: {vector.numel():,} weights")
    tokenizer = SparseVectorTokenizer(
        tokensize=args.token_size,
        device="cpu",
        mode="full_model",
        padding="zero",
        vector_key="vector",
        weight_dimensions_file=str(args.weight_dimensions),
    )
    plot_sparse(vector, tokenizer, args.output)


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

"""Expand the shipped rank-1 LoRA population to the rank-2 flattened layout.

Run once from the repository root:
    python experiments/pad_weights.py

Reads data/all_weights.pt and writes data/all_weights_r1_padded.pt. No layout
file or reference padded population is needed. The hardcoded layout is specific
to this population: each A [1, in] gains a zero row and each B [out, 1] gains a
zero column. Source values and dtype are preserved, including original zeros.

The input and temporary output are memory-mapped and processed in batches.
Saving requires approximately twice the output size in free disk space (about
96 GiB for the shipped float32 population). Existing output files are refused.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_WIDTH = 99648
PADDED_WIDTH = 2 * SOURCE_WIDTH

# Flattening order observed in the shipped population. Each group contains
# three (width, width) A/B pairs, then one (768, width) A/B pair.
GROUP_WIDTHS = (
    320, 320,
    640, 640,
    1280, 1280, 1280, 1280, 1280, 1280,
    640, 640, 640,
    320, 320, 320,
)


def destination_columns() -> list[int]:
    """Map each rank-1 coordinate to its position in the rank-2 vector."""
    columns = []
    offset = 0
    for width in GROUP_WIDTHS:
        for input_width in (width, width, width, 768):
            # A: retain the first row, leaving the second row zero.
            columns.extend(range(offset, offset + input_width))
            offset += 2 * input_width
            # B: retain the first column, leaving the second column zero.
            columns.extend(range(offset, offset + 2 * width, 2))
            offset += 2 * width
    assert len(columns) == SOURCE_WIDTH and offset == PADDED_WIDTH
    return columns


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/all_weights.pt")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/all_weights_r1_padded.pt"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def pad_weights(input_path: Path, output_path: Path, batch_size: int = 32) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    weights = torch.load(input_path, map_location="cpu", weights_only=True, mmap=True)
    if (not isinstance(weights, torch.Tensor) or weights.ndim != 2
            or weights.shape[1] != SOURCE_WIDTH or weights.shape[0] == 0
            or not weights.is_floating_point()):
        raise ValueError(f"Expected a nonempty floating-point tensor [N, {SOURCE_WIDTH}]")

    shape = (weights.shape[0], PADDED_WIDTH)
    size = shape[0] * shape[1]
    columns = torch.tensor(destination_columns(), dtype=torch.int64)
    print(f"Input: {tuple(weights.shape)}, {weights.dtype}")
    print(f"Output: {shape}; temporary + final storage approximately "
          f"{2 * size * weights.element_size() / 2**30:.2f} GiB", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pad-weights-", dir=output_path.parent) as directory:
        padded = torch.from_file(
            str(Path(directory) / "weights.bin"),
            shared=True,
            size=size,
            dtype=weights.dtype,
        ).reshape(shape)
        for start in range(0, shape[0], batch_size):
            batch = padded[start:start + batch_size]
            batch.zero_()
            batch[:, columns] = weights[start:start + batch_size]
            if start // batch_size % 100 == 0:
                print(f"Padded {min(start + batch_size, shape[0])}/{shape[0]} adapters", flush=True)
        print(f"Saving {output_path}", flush=True)
        with output_path.open("xb") as stream:
            torch.save(padded, stream)
        del batch, padded
    print(f"Saved {output_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    pad_weights(args.input, args.output, args.batch_size)


if __name__ == "__main__":
    main()

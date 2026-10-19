"""Utilities for converting flattened LoRA weights into PEFT state dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


WeightDimensions = Mapping[str, Any]


def load_weight_dimensions(path: Path | str) -> WeightDimensions:
    """Load the ordered tensor layout used to flatten a population of LoRAs.

    Dimension files are project artifacts created during dataset preparation. They
    are pickle-backed Torch files and must therefore come from a trusted source.
    """

    dimensions = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(dimensions, Mapping) or not dimensions:
        raise ValueError(f"Expected a non-empty dimensions mapping in {path}")
    return dimensions


def _entry_numel_and_shape(key: str, entry: Any) -> tuple[int, tuple[int, ...]]:
    try:
        numel = int(entry[0][0])
        shape = tuple(int(size) for size in entry[1])
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid dimensions entry for {key!r}: expected [flat_shape, tensor_shape], "
            f"got {entry!r}"
        ) from error

    shape_numel = 1
    for size in shape:
        shape_numel *= size
    if numel != shape_numel:
        raise ValueError(
            f"Invalid dimensions entry for {key!r}: flat size {numel} does not "
            f"match tensor shape {shape} ({shape_numel} values)"
        )
    return numel, shape


def expected_numel(dimensions: WeightDimensions) -> int:
    """Return the number of scalar weights described by ``dimensions``."""

    return sum(
        _entry_numel_and_shape(key, entry)[0]
        for key, entry in dimensions.items()
    )


def _normalize_state_key(key: str) -> str:
    """Normalize legacy w2w keys while leaving bundled PEFT keys unchanged."""

    if key.startswith("lora_unet_"):
        key = key.replace("lora_unet_", "base_model.model.", 1)
        key = key.replace("_lora", ".lora")
        key = key.replace("lora_down", "lora_A")
        key = key.replace("lora_up", "lora_B")

    if not key.endswith((".lora_A.weight", ".lora_B.weight")):
        raise ValueError(
            f"Unsupported LoRA parameter key {key!r}; expected a PEFT key ending "
            "in '.lora_A.weight' or '.lora_B.weight'"
        )
    return key


def flat_weights_to_state_dict(
    weights: torch.Tensor,
    dimensions: WeightDimensions,
) -> dict[str, torch.Tensor]:
    """Split one flat adapter vector into the PEFT tensor layout.

    The insertion order of the dimensions mapping defines the flattening order.
    Returned keys intentionally omit a PEFT adapter name; PEFT's
    ``set_peft_model_state_dict`` inserts the selected name when loading them.
    """

    flat = weights.detach().cpu().flatten()
    expected = expected_numel(dimensions)
    if flat.numel() != expected:
        raise ValueError(
            f"Flat adapter has {flat.numel():,} values, but the dimensions file "
            f"describes {expected:,}"
        )

    state_dict: dict[str, torch.Tensor] = {}
    offset = 0
    for source_key, entry in dimensions.items():
        numel, shape = _entry_numel_and_shape(source_key, entry)
        target_key = _normalize_state_key(source_key)
        if target_key in state_dict:
            raise ValueError(f"Duplicate normalized LoRA key: {target_key}")
        state_dict[target_key] = flat[offset : offset + numel].reshape(shape).contiguous()
        offset += numel

    return state_dict

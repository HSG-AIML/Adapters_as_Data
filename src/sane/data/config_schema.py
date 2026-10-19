import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar

from typing_extensions import Literal



@dataclass
class LoRAWeightsDatasetConfig:
    weights_path: str
    drop_nan: bool
    max_adapters: Optional[int]
    max_absolute_weight: Optional[float]
    rank_label: int = 1

@dataclass
class CombinedLoRAWeightsDatasetConfig:
    """Config for combining rank 1 and rank 2 LoRA datasets."""
    datasets: list[LoRAWeightsDatasetConfig]  # list of individual configs

@dataclass
class TokenizerConfig:
    tokenizer_class: Literal["sparse_vector", "vector"]
    mode: Literal["full_model", "weights_only"]
    tokensize: int
    device: str
    padding: Literal["zero", "mean", "gaussian", "reflect", "replicate", "circular"] = "zero"
    weight_dimensions_file: Optional[str] = None


@dataclass
class CachedWindowedDatasetConfig:
    window_size: int
    num_windows_per_model: int | Literal["auto"]
    pad_windows: bool
    window_distribution_strategy: Literal["consecutive", "distributed"]
    allow_overlapping_windows: bool
    cache_dir: Optional[str]
    data_config_for_hash: dict[str, Any] | None = None
    num_windows_per_layer: int | None = None
    cache_mode: Literal["per_model", "per_window"] = "per_window"
    global_cache: bool = True
    preload: bool = True


@dataclass
class SplitConfig:
    train: float
    val: float
    test: float


@dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    prefetch_factor: int


_T = TypeVar("_T")

import logging

logger = logging.getLogger(__name__)


def from_dict(cls: Type[_T], d: Dict[str, Any], **overrides: Any) -> _T:
    """Construct a dataclass from a dict, ignoring unknown keys."""
    names = {f.name for f in dataclasses.fields(cls)}
    all_keys = {**d, **overrides}
    ignored_keys = [k for k in all_keys if k not in names]
    if ignored_keys:
        logger.debug("Ignoring unknown keys for %s: %s", cls.__name__, ignored_keys)
    return cls(**{k: v for k, v in all_keys.items() if k in names})



@dataclass
class DatasetFactoryConfig:
    checkpoints_dataset: LoRAWeightsDatasetConfig | CombinedLoRAWeightsDatasetConfig
    tokenizer: TokenizerConfig
    tokens_dataset: CachedWindowedDatasetConfig
    split: SplitConfig
    dataloader: DataLoaderConfig

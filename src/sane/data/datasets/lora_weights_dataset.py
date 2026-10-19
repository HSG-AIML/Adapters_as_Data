import torch
import os
from typing import Optional

from sane.data.datasets.checkpoints_dataset import CheckpointsDataset


class LoRAWeightsDataset(CheckpointsDataset):
    """
    Dataset for flattened LoRA adapter weights stored in a single tensor file.

    Uses memory-mapped loading: the weights tensor is paged in from disk on
    access rather than held in RAM. Resident memory stays small even for very
    large datasets, at the cost of an extra disk read per sample (typically
    cached by the OS page cache after the first epoch).

    Expected format of weights_path:
        - torch tensor of shape [N, D] saved with torch.save
        or
        - dict containing a tensor under the key "weights"

    Each item is returned as a 1D torch.Tensor of shape [D].

    Args:
        weights_path: Path to the saved tensor file.
        drop_nan: If True, drop adapters containing any NaN value.
        max_absolute_weight: If set, drop adapters with any |weight| above this.
        dtype: Target dtype ("float32" / "float16" / "bfloat16"). If different
            from the on-disk dtype, mmap is disabled (cast forces materialization).
        max_adapters: If set, use only the first N adapters.
        mmap: Whether to memory-map the file. Default True. Set False to fall
            back to the original load-into-RAM behavior.
        index_cache_path: Optional path to cache the valid-index mask, so that
            filtering doesn't require a full scan on every run. If None, the
            cache is placed next to weights_path.
    """

    def __init__(
        self,
        weights_path: str,
        drop_nan: bool = False,
        max_absolute_weight: Optional[float] = None,
        dtype: Optional[str] = None,
        max_adapters: Optional[int] = None,
        mmap: bool = True,
        index_cache_path: Optional[str] = None,
        rank_label: int = 1
    ):
        super().__init__()

        # Decide whether we can actually use mmap. A dtype cast requires
        # materializing the tensor, so mmap only helps if the on-disk dtype
        # matches the requested dtype (or no cast is requested).
        use_mmap = mmap

        print("LoRAWeightsDataset: starting torch.load (mmap={})".format(use_mmap))
        if use_mmap:
            # mmap=True requires weights_only=True in recent PyTorch versions
            try:
                obj = torch.load(weights_path, map_location="cpu", mmap=True, weights_only=True)
            except TypeError:
                # Older PyTorch without mmap support — fall back gracefully
                print("LoRAWeightsDataset: torch.load mmap not supported, falling back to full load")
                obj = torch.load(weights_path, map_location="cpu")
                use_mmap = False
        else:
            obj = torch.load(weights_path, map_location="cpu")
        print("LoRAWeightsDataset: finished torch.load", type(obj), getattr(obj, "shape", None))

        if isinstance(obj, torch.Tensor):
            weights = obj
        elif isinstance(obj, dict):
            if "weights" in obj:
                weights = obj["weights"]
            else:
                raise ValueError(
                    f"Unsupported dict format in {weights_path}. "
                    f"Available keys: {list(obj.keys())}"
                )
        else:
            raise ValueError(
                f"Unsupported object type in {weights_path}: {type(obj)}"
            )

        if weights.ndim != 2:
            raise ValueError(
                f"Expected weights tensor with shape [N, D], got {tuple(weights.shape)}"
            )

        # Dtype handling: only cast if on-disk dtype differs from requested dtype.
        # Any cast disables mmap because it materializes the full tensor.
        if dtype is not None:
            target_dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(dtype)
            if target_dtype is None:
                raise ValueError(f"Unsupported dtype: {dtype}")
            if weights.dtype != target_dtype:
                print(f"LoRAWeightsDataset: casting {weights.dtype} -> {target_dtype} (disables mmap)")
                weights = weights.to(target_dtype)
                use_mmap = False

        print("before filtering: shape", weights.shape)

        if max_adapters is not None:
            # Slicing a mmap'd tensor returns a view; this is cheap.
            weights = weights[:max_adapters]

        # Filtering. A naive scan touches every page and defeats mmap.
        # Cache the valid-index list to disk so subsequent runs are fast.
        needs_filtering = drop_nan or (max_absolute_weight is not None)
        if needs_filtering:
            cache_path = index_cache_path or self._default_cache_path(
                weights_path, max_adapters, drop_nan, max_absolute_weight
            )
            valid_indices = self._load_or_compute_valid_indices(
                weights, cache_path, drop_nan, max_absolute_weight
            )
            print("after filtering: kept {} / {} adapters".format(
                len(valid_indices), weights.shape[0]
            ))
        else:
            valid_indices = torch.arange(weights.shape[0])

        self.weights = weights
        self.valid_indices = valid_indices
        self.use_mmap = use_mmap
        self.weights_path = weights_path
        self.rank_label = rank_label

        print("LoRAWeightsDataset: final shape", (len(self.valid_indices), weights.shape[1]))

    @staticmethod
    def _default_cache_path(
        weights_path: str,
        max_adapters: Optional[int],
        drop_nan: bool,
        max_absolute_weight: Optional[float],
    ) -> str:
        """Build a cache path encoding the filter parameters."""
        base = os.path.splitext(weights_path)[0]
        tag = "valid_idx"
        if max_adapters is not None:
            tag += f"_max{max_adapters}"
        if drop_nan:
            tag += "_nan"
        if max_absolute_weight is not None:
            tag += f"_abs{max_absolute_weight}"
        return f"{base}.{tag}.pt"

    @staticmethod
    def _load_or_compute_valid_indices(
        weights: torch.Tensor,
        cache_path: str,
        drop_nan: bool,
        max_absolute_weight: Optional[float],
    ) -> torch.Tensor:
        """Load valid indices from cache, or compute and cache them."""
        if os.path.exists(cache_path):
            print(f"LoRAWeightsDataset: loading cached valid indices from {cache_path}")
            return torch.load(cache_path, map_location="cpu")

        print(f"LoRAWeightsDataset: computing valid indices (full scan, one-time cost)")
        # Process in chunks to keep peak memory low even when scanning a mmap'd tensor
        chunk_size = 256
        masks = []
        for start in range(0, weights.shape[0], chunk_size):
            end = min(start + chunk_size, weights.shape[0])
            chunk = weights[start:end]
            mask = torch.ones(chunk.shape[0], dtype=torch.bool)
            if drop_nan:
                mask &= ~torch.isnan(chunk).any(dim=1)
            if max_absolute_weight is not None:
                mask &= chunk.abs().amax(dim=1) <= max_absolute_weight
            masks.append(mask)
        full_mask = torch.cat(masks)
        valid_indices = torch.nonzero(full_mask, as_tuple=False).squeeze(1)

        try:
            torch.save(valid_indices, cache_path)
            print(f"LoRAWeightsDataset: cached valid indices to {cache_path}")
        except OSError as e:
            print(f"LoRAWeightsDataset: failed to cache valid indices: {e}")

        return valid_indices

    def __getitem__(self, index: int) -> dict:
        real_index = int(self.valid_indices[index])
        # .clone() forces a copy out of the mmap'd region into normal memory,
        # which is what we want — the returned tensor is then independent and
        # safe to pin / transfer to GPU.
        return {"vector": self.weights[real_index].clone(),
                "rank":   torch.tensor(self.rank_label, dtype=torch.long)
                }

    def __len__(self) -> int:
        return len(self.valid_indices)
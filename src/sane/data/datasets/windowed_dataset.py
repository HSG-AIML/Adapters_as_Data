import logging
from collections.abc import Iterable
from typing import List, Literal, Tuple

import torch
from torch.utils.data import Dataset

from sane.data.datasets.checkpoints_dataset import Checkpoint, CheckpointsDataset
from sane.data.tokenizers import SANETokenizer

logger = logging.getLogger(__name__)

# Type alias. Future improvement: Explicit Dataclass and custom collate for this.
Window = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class WindowedDataset(Dataset):
    """Base class for datasets that return one window per __getitem__ call.

    Owns all logic that is independent of how windows are stored or retrieved:
    window extraction algorithms, tokenization, augmentation stacking,
    index mapping, and length computation.

    While this can be used directly, it will reload the checkpoint every time from scratch
    to provide a window, which may prove inefficient.
    For better performance, use the cached version of this class.

    Args:
        checkpoints_dataset: The underlying dataset of checkpoints.
        tokenizer: Tokenizer used to convert state dicts to token sequences.
        window_size: Number of tokens per window.
        num_windows_per_model: Windows to sample per checkpoint. Set to "auto" for full
            coverage of all tokens in the checkpoint.
        num_windows_per_layer: Windows to sample per layer. When specified, takes
            precedence over num_windows_per_model. Cannot be used with tokenizer mode
            'full_model'.
        pad_windows: If True, pad short windows with zeros to exactly window_size tokens.
        allow_overlapping_windows: If True, allow windows to overlap when placing them
            within a layer. If False, windows are placed consecutively without overlap.
        window_distribution_strategy: Strategy for placing non-overlapping windows.
            "consecutive" places windows sequentially from the start of the layer.
            "distributed" spreads windows evenly across the layer.
        **kwargs: Additional keyword arguments are ignored (with a debug log message).
    """

    def __init__(
        self,
        checkpoints_dataset: CheckpointsDataset,
        tokenizer: SANETokenizer,
        window_size: int,
        num_windows_per_model: int | Literal["auto"] = 1,
        num_windows_per_layer: int | None = None,
        pad_windows: bool = False,
        allow_overlapping_windows: bool = False,
        window_distribution_strategy: Literal[
            "consecutive", "distributed"
        ] = "consecutive",
        **kwargs,
    ):
        if kwargs:
            logger.debug(
                f"{type(self).__name__} received unknown kwargs (ignored): {list(kwargs)}"
            )

        self.checkpoints_dataset: CheckpointsDataset = checkpoints_dataset
        self.tokenizer: SANETokenizer = tokenizer
        self.tokensize: int = tokenizer.tokensize
        self.window_size: int = window_size
        self.num_windows_per_model: int | Literal["auto"] = num_windows_per_model
        self.num_windows_per_layer: int | None = num_windows_per_layer
        self.pad_windows: bool = pad_windows
        self.allow_overlapping_windows: bool = allow_overlapping_windows
        self.window_distribution_strategy: Literal["consecutive", "distributed"] = (
            window_distribution_strategy
        )

        if num_windows_per_layer is not None and num_windows_per_model != "auto":
            logger.warning(
                "num_windows_per_layer and num_windows_per_model are both specified; "
                "per-layer windowing will take precedence."
            )

        if num_windows_per_layer is not None and tokenizer.mode == "full_model":
            raise ValueError(
                "Per-layer windowing (num_windows_per_layer) cannot be used with tokenizer "
                "mode 'full_model'. Use num_windows_per_model instead."
            )

        self._windows_per_checkpoint = self._calculate_windows_per_checkpoint()

    def _get_window(self, checkpoint_idx: int, window_idx: int) -> Window:
        """Return a single window, generated on-the-fly. Override to add caching."""
        return self._generate_windows(checkpoint_idx)[window_idx]

    def _get_all_windows(self, checkpoint_idx: int) -> List[Window]:
        """Return all windows for a checkpoint, generated on-the-fly. Override to add caching."""
        return self._generate_windows(checkpoint_idx)

    def __len__(self) -> int:
        return len(self.checkpoints_dataset) * self._windows_per_checkpoint

    def __len_models__(self) -> int:
        return len(self.checkpoints_dataset)

    def __getitem__(self, index: int) -> Window:
        checkpoint_idx, window_idx = self._map_index(index)
        return self._get_window(checkpoint_idx, window_idx)

    def __getmodel__(self, index: int) -> List[Window]:
        if index >= len(self.checkpoints_dataset):
            raise IndexError(
                f"Checkpoint index {index} out of range for dataset of size "
                f"{len(self.checkpoints_dataset)}"
            )
        return self._get_all_windows(index)

    def _map_index(self, index: int) -> Tuple[int, int]:
        if index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self)}"
            )
        return divmod(index, self._windows_per_checkpoint)

    def _generate_windows(self, checkpoint_idx: int) -> List[Window]:
        """Load a checkpoint (handling augmentation views) and return all windows."""
        checkpoints = self.checkpoints_dataset[checkpoint_idx]

        # single sample: normal checkpoint or LoRA dict
        if isinstance(checkpoints, Checkpoint) or isinstance(checkpoints, dict):
            # no augmentations present
            return self._tokenize_checkpoint(checkpoints)

        # augmentations: tokenize all variations
        tokenized_checkpoints = [
            self._tokenize_checkpoint(ckpt) for ckpt in checkpoints
        ]

        if not all(
            len(tc) == len(tokenized_checkpoints[0]) for tc in tokenized_checkpoints
        ):
            raise ValueError(
                "All augmentation views must produce the same number of windows."
            )

        return [
            tuple(torch.stack(tensors, dim=0) for tensors in zip(*windows))
            for windows in zip(*tokenized_checkpoints)
        ]  # type: ignore

    def _tokenize_checkpoint(self, checkpoint) -> List[Window]:
        """Tokenize a single checkpoint and apply windowing."""
        if isinstance(checkpoint, Checkpoint):
            tokenized_result = self.tokenizer.tokenize(checkpoint.model.state_dict())
        else:
            tokenized_result = self.tokenizer.tokenize(checkpoint)

        if isinstance(tokenized_result, tuple):
            layer_data = [tokenized_result]
        else:
            layer_data = tokenized_result

        return self._extract_windows_from_layers(layer_data)

    def _get_sample_layer_data(self) -> list[Window]:
        """Tokenize the first checkpoint to probe layer structure."""
        if len(self.checkpoints_dataset) == 0:
            raise ValueError("Checkpoints dataset is empty.")

        checkpoint = self.checkpoints_dataset[0]
        if isinstance(checkpoint, Iterable) and not isinstance(checkpoint, Checkpoint):
            checkpoint = checkpoint[0]

        tokenized_result = self.tokenizer.tokenize(checkpoint.model.state_dict())
        return (
            [tokenized_result]
            if isinstance(tokenized_result, tuple)
            else tokenized_result
        )

    def _calculate_windows_per_checkpoint(self) -> int:
        """Return the number of windows produced per checkpoint."""
        if (
            isinstance(self.num_windows_per_model, int)
            and self.num_windows_per_layer is None
        ):
            return self.num_windows_per_model

        layer_data = self._get_sample_layer_data()
        windows = self._extract_windows_from_layers(layer_data)
        total = len(windows)
        logger.info(
            f"Estimated windows per checkpoint: {total} (from {len(layer_data)} layers)"
        )
        return total

    def _extract_windows_from_layers(
        self,
        layer_data: List[Window],
    ) -> List[Window]:
        if self.num_windows_per_layer is not None:
            return self._apply_per_layer_windowing_to_layers(layer_data)
        return self._apply_regular_windowing_to_layers(layer_data)

    def _apply_regular_windowing_to_layers(
        self,
        layer_data: List[Window],
    ) -> List[Window]:
        if self.num_windows_per_model == "auto":
            windows = []
            for layer_tokens, layer_mask, layer_position in layer_data:
                layer_length = layer_tokens.shape[0]
                if self.pad_windows:
                    n = max(
                        1, (layer_length + self.window_size - 1) // self.window_size
                    )
                else:
                    n = max(1, layer_length // self.window_size)
                windows.extend(
                    self._sample_windows_from_layer(
                        layer_tokens, layer_mask, layer_position, n
                    )
                )
            return windows

        if not isinstance(self.num_windows_per_model, int):
            raise ValueError(
                f"Expected int for num_windows_per_model, got {type(self.num_windows_per_model)}"
            )

        total_tokens = sum(t.shape[0] for t, _, _ in layer_data)
        windows = []
        remaining = self.num_windows_per_model

        for i, (layer_tokens, layer_mask, layer_position) in enumerate(layer_data):
            if i == len(layer_data) - 1:
                layer_windows = remaining
            else:
                proportion = layer_tokens.shape[0] / total_tokens
                layer_windows = min(
                    max(1, int(self.num_windows_per_model * proportion)),
                    remaining,
                )

            if layer_windows > 0:
                windows.extend(
                    self._sample_windows_from_layer(
                        layer_tokens, layer_mask, layer_position, layer_windows
                    )
                )
                remaining -= layer_windows

            if remaining <= 0:
                break

        if len(windows) < self.num_windows_per_model:
            logger.warning(
                f"Requested {self.num_windows_per_model} windows but only {len(windows)} could be sampled."
            )

        return windows

    def _apply_per_layer_windowing_to_layers(
        self,
        layer_data: List[Window],
    ) -> List[Window]:
        windows = []
        for layer_tokens, layer_mask, layer_position in layer_data:
            windows.extend(
                self._sample_windows_from_layer(
                    layer_tokens,
                    layer_mask,
                    layer_position,
                    self.num_windows_per_layer or 1,
                )
            )
        return windows

    def _sample_windows_from_layer(
        self,
        layer_tokens: torch.Tensor,
        layer_mask: torch.Tensor,
        layer_position: torch.Tensor,
        num_windows: int,
    ) -> List[Window]:
        layer_length = layer_tokens.shape[0]
        windows = []

        if layer_length <= self.window_size:
            windows.append(
                self._pad_window_if_needed((layer_tokens, layer_mask, layer_position))
            )
        else:
            for start in self._get_window_starts(layer_length, num_windows):
                end = min(start + self.window_size, layer_length)
                window = (
                    layer_tokens[start:end],
                    layer_mask[start:end],
                    layer_position[start:end].clone(),
                )
                windows.append(self._pad_window_if_needed(window))

        if len(windows) != num_windows:
            size_desc = (
                f"of size {self.window_size}"
                if self.pad_windows
                else f"of size ≤{self.window_size}"
            )
            logger.warning(
                f"Requested {num_windows} windows but only {len(windows)} {size_desc} "
                f"could be sampled from layer of length {layer_length}."
            )

        return windows

    def _pad_window_if_needed(
        self,
        window_tuple: Window,
    ) -> Window:
        tokens, mask, position = window_tuple
        current_length = tokens.shape[0]

        if not self.pad_windows or current_length >= self.window_size:
            return window_tuple

        pad = self.window_size - current_length

        pad_tokens = torch.zeros(
            (pad, self.tokensize), dtype=tokens.dtype, device=tokens.device
        )
        pad_mask = torch.zeros(
            (pad, self.tokensize), dtype=torch.bool, device=mask.device
        )

        if current_length > 0:
            last = position[-1]
            pad_position = torch.tensor(
                [
                    [last[0].item() + i + 1, last[1].item(), last[2].item() + i + 1]
                    for i in range(pad)
                ],
                dtype=position.dtype,
                device=position.device,
            )
        else:
            pad_position = torch.zeros(
                (pad, 3), dtype=torch.int32, device=position.device
            )

        return (
            torch.cat([tokens, pad_tokens], dim=0),
            torch.cat([mask, pad_mask], dim=0),
            torch.cat([position, pad_position], dim=0),
        )

    def _get_window_starts(self, total_tokens: int, num_windows: int) -> List[int]:
        if total_tokens <= self.window_size:
            return [0]

        max_start = total_tokens - self.window_size

        if num_windows == 1:
            if self.allow_overlapping_windows:
                return [int(torch.randint(0, max_start + 1, (1,)).item())]
            return [0]

        if not self.allow_overlapping_windows:
            if self.pad_windows:
                max_non_overlapping = (
                    total_tokens + self.window_size - 1
                ) // self.window_size
            else:
                max_non_overlapping = (max_start // self.window_size) + 1
            actual = min(num_windows, max_non_overlapping)

            if self.window_distribution_strategy == "distributed":
                starts = self._get_distributed_window_starts(total_tokens, actual)
            else:
                starts = [i * self.window_size for i in range(actual)]

            if actual < num_windows:
                import warnings

                warnings.warn(
                    f"Requested {num_windows} non-overlapping windows of size {self.window_size} "
                    f"but only {actual} fit in sequence of length {total_tokens}."
                )
            return starts

        if num_windows >= max_start + 1:
            return list(range(0, max_start + 1, self.window_size))

        starts: List[int] = torch.randperm(max_start + 1)[:num_windows].tolist()
        return sorted(starts)

    def _get_distributed_window_starts(
        self, total_tokens: int, num_windows: int
    ) -> List[int]:
        if num_windows <= 1:
            return [0]

        max_start = total_tokens - self.window_size

        if num_windows == 2:
            return [0, max_start]

        step = max_start // (num_windows - 1)
        starts = [i * step for i in range(num_windows)]

        final: List[int] = []
        for s in starts:
            while final and s < final[-1] + self.window_size:
                s = final[-1] + self.window_size
            if s <= max_start:
                final.append(s)
            else:
                break

        return final

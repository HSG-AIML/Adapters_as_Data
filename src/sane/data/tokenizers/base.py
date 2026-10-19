from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

StateDict = dict[str, Any]

logger = logging.getLogger(__name__)

class SANETokenizer(nn.Module, ABC):
    """Abstract base for SANE tokenizers."""

    def __init__(
        self,
        *,
        tokensize: int = 230,
        device: str = "cpu",
        mode: str = "layer_wise",
        reference_statedict: StateDict | None = None,
        padding: str = "zero",
    ):
        super().__init__()
        if mode not in ("layer_wise", "full_model"):
            raise ValueError(f"mode must be 'layer_wise' or 'full_model', got {mode}")
        self.tokensize = tokensize
        self.device = device
        self.mode = mode
        self.reference_statedict = reference_statedict
        self.forward = self.tokenize
        self.padding = padding

    def tokenize(
        self, statedict: StateDict
    ) -> Union[
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Tokenizes the input checkpoint according to the specified mode.

        Args:
            statedict (StateDict): The input checkpoint to tokenize.

        Returns:
            Union[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                Each tuple contains:
                - tokens (torch.Tensor): The tokenized weights.
                - mask (torch.Tensor): A mask indicating the valid tokens.
                - position (torch.Tensor): The positions of the tokens.

                If mode is "layer_wise": Returns a list of tuples, one per layer.
                If mode is "full_model": Returns a single tuple (all layers concatenated).
        """
        flattened = self.flatten(statedict)
        if self.mode == "layer_wise":
            return self.slice_by_layers(flattened)
        return self.slice(flattened)

    def slice(self, flattened) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slices the flattened weights into tokens, mask, and position tensors.

        Concatenates layer-wise sliced tokens into full sequence.

        Args:
            flattened (List[torch.Tensor]): The flattened weights to slice.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - tokens (torch.Tensor): The sliced tokens (all layers concatenated).
                - mask (torch.Tensor): A mask indicating the valid tokens (all layers concatenated).
                - position (torch.Tensor): The positions of the tokens (all layers concatenated).
        """
        layerwise = self.slice_by_layers(flattened)
        tokens = torch.cat([t for (t, _, _) in layerwise], dim=0)
        mask = torch.cat([m for (_, m, _) in layerwise], dim=0)
        pos = torch.cat([p for (_, _, p) in layerwise], dim=0)
        return tokens, mask, pos

    def detokenize(
        self, tokens_input, mask=None, position=None, reference_statedict=None
    ) -> StateDict:
        """Detokenizes the input tokens by unslicing and rebuilding the state dictionary.

        Args:
            tokens_input (Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Either a single tensor (full model format) or a list of tuples (layer-wise format).
            mask (torch.Tensor, optional): The mask indicating the valid tokens. Required for full model format.
            position (torch.Tensor, optional): The positions of the tokens. Required for full model format.
            reference_statedict (StateDict, optional): A reference state dictionary to use for rebuilding. Default is None.

        Returns:
            StateDict: The detokenized state dictionary.
        """
        flat = self.unslice(tokens_input, mask, position)
        return self.rebuild_state_dict(flat, reference_statedict)

    def pad(self, w: torch.Tensor, pad_right: int) -> torch.Tensor:
        """Pads a tensor along its last dimension by pad_right elements.

        Supports the following modes:
            - 'zero': Pads each token with zeros.
            - 'mean': Pads each token with the mean value of its valid elements.
            - 'gaussian': Pads each token with Gaussian noise based on the mean and std of its valid elements.
            - 'reflect': Pads by reflecting the valid elements (requires at least 2 valid elements).
            - 'replicate': Pads by replicating the edge valid element.
            - 'circular': Pads by wrapping around the valid elements.

        For 'reflect', 'replicate', and 'circular' modes, iteratively pads to handle
        cases where pad_right exceeds the valid reflection/replication size.

        Args:
            w (torch.Tensor): The input tensor to pad.
            pad_right (int): The number of elements to pad on the right side.

        Returns:
            torch.Tensor: The padded tensor.
        """
        if pad_right == 0:
            return w

        # Padding requires at least one (or two for reflect) valid element(s)
        if w.shape[-1] < 1:
            raise ValueError(f"Input tensor has no valid elements to pad (size {w.shape[-1]}), cannot apply padding mode '{self.padding}'.")
        elif self.padding == "reflect" and w.shape[-1] < 2:
            # Fallback to zero padding if input to reflect padding is too small
            logger.warning(f"Input size {w.shape[-1]} is too small for padding mode '{self.padding}'. Falling back to zero padding.")
            return F.pad(w, (0, pad_right), mode='constant', value=0)

        if self.padding == "zero":
            w_pad = F.pad(w, (0, pad_right), mode='constant', value=0)

        elif self.padding == "mean":
            # Compute mean per token
            mean_per_token = w.mean(dim=-1, keepdim=True)

            # Expand mean to the right and concatenate with original tensor
            mean_padding = mean_per_token.expand(*w.shape[:-1], pad_right)
            w_pad = torch.cat([w, mean_padding], dim=-1)

        elif self.padding == "gaussian":
            # Compute mean and std per token
            mean_per_token = w.mean(dim=-1, keepdim=True)
            std_per_token = w.std(dim=-1, keepdim=True, unbiased=False)

            # Sample Gaussian noise and concatenate with original tensor
            gaussian_padding = (
                torch.randn(*w.shape[:-1], pad_right, device=w.device, dtype=w.dtype)
                * std_per_token
                + mean_per_token
            )
            w_pad = torch.cat([w, gaussian_padding], dim=-1)

        elif self.padding in ("reflect", "replicate", "circular"):

            # Non-constant PyTorch padding modes require at least 2D input
            needs_squeeze = w.dim() == 1
            if needs_squeeze:
                w = w.unsqueeze(0)

            # Iteratively pad until we've added all required padding (handles large pad_right)
            w_pad = w
            remaining = pad_right
            while remaining > 0:
                max_step = w_pad.shape[-1] - (1 if self.padding == "reflect" else 0)
                step = min(remaining, max_step)
                w_pad = F.pad(w_pad, (0, step), mode=self.padding)
                remaining -= step

            # Remove the extra dimension if we added one for padding
            if needs_squeeze:
                w_pad = w_pad.squeeze(0)

        else:
            raise ValueError(
                f"Unsupported padding mode '{self.padding}'. "
                "Expected one of: zero, mean, gaussian, reflect, replicate, circular."
            )

        return w_pad

    @abstractmethod
    def flatten(self, statedict: StateDict) -> List[torch.Tensor]: ...
    @abstractmethod
    def slice_by_layers(
        self, flattened_weights
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: ...
    @abstractmethod
    def unslice(self, tokens_input, mask=None, position=None) -> List[torch.Tensor]: ...
    @abstractmethod
    def rebuild_state_dict(
        self, flattened_weights, reference_statedict=None
    ) -> StateDict: ...

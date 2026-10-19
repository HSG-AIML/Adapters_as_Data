from typing import Any, List, Tuple

import torch

from sane.data.tokenizers.base import SANETokenizer

StateDict = dict[str, Any]


class VectorTokenizer(SANETokenizer):
    """
    Tokenizer for pre-flattened weight vectors.

    Expected input state dict:
        {
            "vector": torch.Tensor of shape [D]
        }

    The vector is treated as a single logical layer and chunked into tokens
    of size `tokensize`.

    Output follows the standard SANE tokenizer convention:
        tokens:   [num_tokens, tokensize]
        mask:     [num_tokens, tokensize]  (True for valid elements, False for padding)
        position: [num_tokens, 3]

    Position encoding convention used here:
        position[:, 0] = 0             (constant; kept for compatibility)
        position[:, 1] = layer index   (always 0 for vector tokenizer)
        position[:, 2] = token index within that layer
    """

    def __init__(
        self,
        *,
        tokensize: int = 230,
        device: str = "cpu",
        mode: str = "full_model",
        reference_statedict: StateDict | None = None,
        padding: str = "zero",
        vector_key: str = "vector",
    ):
        super().__init__(
            tokensize=tokensize,
            device=device,
            mode=mode,
            reference_statedict=reference_statedict,
            padding=padding,
        )
        self.vector_key = vector_key

    def flatten(self, statedict: StateDict) -> List[torch.Tensor]:
        """
        Extract the vector from the input state dict and return it as a
        single-element list to match the base tokenizer contract.
        """
        if not isinstance(statedict, dict):
            raise TypeError(f"Expected statedict to be dict, got {type(statedict)}")

        if self.vector_key not in statedict:
            raise KeyError(
                f"Expected key '{self.vector_key}' in statedict, got keys {list(statedict.keys())}"
            )

        vector = statedict[self.vector_key]

        if not isinstance(vector, torch.Tensor):
            raise TypeError(
                f"Expected statedict['{self.vector_key}'] to be torch.Tensor, got {type(vector)}"
            )

        vector = vector.detach().to(self.device).flatten()

        if vector.ndim != 1:
            raise ValueError(
                f"Expected flattened vector to be 1D, got shape {tuple(vector.shape)}"
            )

        return [vector]

    def slice_by_layers(
        self, flattened_weights: List[torch.Tensor]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Slice each flattened tensor into fixed-size tokens.

        Since this tokenizer treats the full vector as a single layer,
        `flattened_weights` is expected to contain exactly one tensor.
        """
        outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for layer_idx, w in enumerate(flattened_weights):
            if w.ndim != 1:
                raise ValueError(
                    f"Expected each flattened weight tensor to be 1D, got shape {tuple(w.shape)}"
                )

            n_values = w.numel()
            if n_values == 0:
                raise ValueError("Cannot tokenize an empty vector.")

            remainder = n_values % self.tokensize
            pad_right = 0 if remainder == 0 else self.tokensize - remainder

            w_pad = self.pad(w, pad_right)
            n_tokens = w_pad.numel() // self.tokensize

            tokens = w_pad.view(n_tokens, self.tokensize)

            mask = torch.ones((n_tokens, self.tokensize), dtype=torch.bool, device=tokens.device)
            if pad_right > 0:
                mask[-1, -pad_right:] = False

            position = torch.stack(
                [
                    torch.zeros(n_tokens, dtype=torch.long, device=tokens.device),
                    torch.full((n_tokens,), layer_idx, dtype=torch.long, device=tokens.device),
                    torch.arange(n_tokens, dtype=torch.long, device=tokens.device),
                ],
                dim=1,
            )

            # position = torch.stack(
            #     [
            #         torch.full((n_tokens,), layer_idx, dtype=torch.long, device=tokens.device),
            #         torch.arange(n_tokens, dtype=torch.long, device=tokens.device),
            #     ],
            #     dim=1,
            # )


            outputs.append((tokens, mask, position))

        return outputs

    def unslice(self, tokens_input, mask=None, position=None) -> List[torch.Tensor]:
        """
        Reconstruct flattened vectors from tokenized representation.

        Supports:
        - full_model format: tokens_input, mask, position
        - layer_wise format: list of (tokens, mask, position)
        
        Position can be 2D [num_tokens, 2] (legacy) or 3D [num_tokens, 3] (current).
        In 2D: position[:, 0] = layer_id, position[:, 1] = token_idx
        In 3D: position[:, 0] = dummy zero, position[:, 1] = layer_id, position[:, 2] = token_idx
        """
        if isinstance(tokens_input, list):
            # layer-wise mode: list of tuples
            flattened_weights: List[torch.Tensor] = []
            for item in tokens_input:
                if not isinstance(item, tuple) or len(item) != 3:
                    raise ValueError(
                        "For layer-wise format, expected a list of (tokens, mask, position) tuples."
                    )
                tokens, layer_mask, _layer_position = item
                flat = tokens.reshape(-1)[layer_mask.reshape(-1)]
                flattened_weights.append(flat)
            return flattened_weights

        # full-model mode
        if mask is None or position is None:
            raise ValueError(
                "For full_model format, mask and position must be provided."
            )

        if tokens_input.ndim != 2:
            raise ValueError(
                f"Expected tokens_input to have shape [num_tokens, tokensize], got {tuple(tokens_input.shape)}"
            )

        if mask.shape != tokens_input.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match tokens_input shape {tuple(tokens_input.shape)}"
            )

        if position.ndim != 2 or position.shape[1] not in (2, 3):
            raise ValueError(
                f"Expected position shape [num_tokens, 2] or [num_tokens, 3], got {tuple(position.shape)}"
            )

        # Layer id column: 0 for legacy 2D positions, 1 for current 3D positions
        layer_id_col = 0 if position.shape[1] == 2 else 1

        layer_ids = torch.unique(position[:, layer_id_col], sorted=True)
        flattened_weights = []

        for layer_id in layer_ids:
            layer_selector = position[:, layer_id_col] == layer_id
            layer_tokens = tokens_input[layer_selector]
            layer_mask = mask[layer_selector]
            flat = layer_tokens.reshape(-1)[layer_mask.reshape(-1)]
            flattened_weights.append(flat)

        return flattened_weights

    def rebuild_state_dict(
        self, flattened_weights: List[torch.Tensor], reference_statedict=None
    ) -> StateDict:
        """
        Rebuild a state dict from flattened weights.

        If a reference state dict is provided and contains the vector key,
        the reconstructed vector is reshaped to the reference shape.
        """
        if len(flattened_weights) != 1:
            raise ValueError(
                f"VectorTokenizer expects exactly 1 flattened tensor, got {len(flattened_weights)}"
            )

        vector = flattened_weights[0]

        ref = reference_statedict if reference_statedict is not None else self.reference_statedict
        if ref is not None and self.vector_key in ref:
            ref_tensor = ref[self.vector_key]
            if not isinstance(ref_tensor, torch.Tensor):
                raise TypeError(
                    f"reference_statedict['{self.vector_key}'] must be a torch.Tensor, got {type(ref_tensor)}"
                )
            vector = vector.view_as(ref_tensor)

        return {self.vector_key: vector}
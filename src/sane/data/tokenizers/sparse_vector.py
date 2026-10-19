from typing import Any, List, Tuple

import torch

from sane.data.tokenizers.base import SANETokenizer

StateDict = dict[str, Any]

# A weight_dimensions dict maps each LoRA matrix key to a
# [flat_shape, real_shape] pair, e.g.
#   {"...attn1.to_q.lora_A.weight": [torch.Size([320]), torch.Size([1, 320])]}
# The segment size is flat_shape[0]; real_shape is used to reshape on rebuild.
WeightDimensions = dict[str, Any]


class SparseVectorTokenizer(SANETokenizer):
    """
    Tokenizer for pre-flattened weight vectors that slices according to the
    underlying weight-matrix boundaries rather than a single contiguous stream.

    Motivation:
        A plain ``VectorTokenizer`` chunks the whole vector into fixed-size
        tokens. When matrices of differing sizes are concatenated (e.g. LoRA
        ``to_q`` of dim 320/640/1280 alongside cross-attention ``to_v`` of dim
        768), a fixed chunk size that is not a common multiple causes tokens to
        straddle two distinct matrices. This tokenizer instead respects each
        matrix's boundary: a matrix is split into ``ceil(numel / tokensize)``
        tokens, and only the *last* token of each matrix is padded. No token
        ever mixes weights from two different matrices.

    Expected input state dict:
        {
            "vector": torch.Tensor of shape [D]
        }
    where ``D == sum(flat_shape[0] for every entry in weight_dimensions)``.

    Boundaries come from a ``weight_dimensions`` dict of the form
    ``{key: [flat_shape, real_shape]}`` (see document 2 in the design notes).
    The flat vector is consumed in that dict's insertion order, the segment
    size for each key is ``flat_shape[0]``, and ``real_shape`` is used to
    reshape each segment when rebuilding. The dict can be passed directly via
    ``weight_dimensions`` or loaded from disk via ``weight_dimensions_file``
    (``torch.load(..., map_location="cpu")``). This mirrors the reference
    ``unflatten`` routine:

        counter = 0
        for key in weight_dimensions:
            final[key] = flat[counter:counter + dims[key][0][0]].unflatten(0, dims[key][1])
            counter += dims[key][0][0]

    Output follows the standard SANE tokenizer convention:
        tokens:   [num_tokens, tokensize]
        mask:     [num_tokens, tokensize]  (True for valid, False for padding)
        position: [num_tokens, 3]

    Position encoding convention (matching the layer-major stack the base
    VectorTokenizer emits):
        position[:, 0] = 0                (constant; kept for compatibility)
        position[:, 1] = matrix index     (one per weight_dimensions entry)
        position[:, 2] = token index within that matrix
    """

    def __init__(
        self,
        *,
        tokensize: int = 768,
        device: str = "cpu",
        mode: str = "full_model",
        reference_statedict: StateDict | None = None,
        padding: str = "zero",
        vector_key: str = "vector",
        weight_dimensions: WeightDimensions | None = None,
        weight_dimensions_file: str | None = None,
    ):
        super().__init__(
            tokensize=tokensize,
            device=device,
            mode=mode,
            reference_statedict=reference_statedict,
            padding=padding,
        )
        self.vector_key = vector_key
        self.weight_dimensions_file = weight_dimensions_file

        print(f"weight_dimensions: {weight_dimensions}")
        print(f"weight_dimensions_file: {weight_dimensions_file}")

        # An explicit dict takes precedence; otherwise load from file if given.
        if weight_dimensions is None and weight_dimensions_file is not None:
            weight_dimensions = torch.load(
                weight_dimensions_file, map_location="cpu"
            )
        self.weight_dimensions = weight_dimensions

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _resolve_dims(
        self, weight_dimensions: WeightDimensions | None
    ) -> WeightDimensions:
        dims = (
            weight_dimensions
            if weight_dimensions is not None
            else self.weight_dimensions
        )
        if dims is None:
            raise ValueError(
                "SparseVectorTokenizer requires a weight_dimensions dict to "
                "infer matrix boundaries (none provided at init or call time)."
            )
        if not isinstance(dims, dict) or len(dims) == 0:
            raise ValueError(
                "weight_dimensions must be a non-empty dict of "
                "{key: [flat_shape, real_shape]}."
            )
        return dims

    def _segment_sizes(
        self, weight_dimensions: WeightDimensions | None
    ) -> List[int]:
        """
        Derive the per-matrix element counts from a weight_dimensions dict, in
        insertion order. The size of each matrix is ``flat_shape[0]``, i.e.
        ``weight_dimensions[key][0][0]``.
        """
        dims = self._resolve_dims(weight_dimensions)
        sizes: List[int] = []
        for key, value in dims.items():
            try:
                size = int(value[0][0])
            except (TypeError, IndexError) as exc:
                raise ValueError(
                    f"weight_dimensions['{key}'] is malformed; expected "
                    f"[flat_shape, real_shape], got {value!r}"
                ) from exc
            sizes.append(size)
        return sizes

    def _segment_real_shapes(
        self, weight_dimensions: WeightDimensions | None
    ) -> List[Any]:
        """Per-matrix real (target) shapes from ``weight_dimensions[key][1]``."""
        dims = self._resolve_dims(weight_dimensions)
        shapes: List[Any] = []
        for key, value in dims.items():
            try:
                shapes.append(value[1])
            except (TypeError, IndexError) as exc:
                raise ValueError(
                    f"weight_dimensions['{key}'] is malformed; expected "
                    f"[flat_shape, real_shape], got {value!r}"
                ) from exc
        return shapes

    # ------------------------------------------------------------------ #
    # flatten
    # ------------------------------------------------------------------ #
    def flatten(self, statedict: StateDict) -> List[torch.Tensor]:
        """
        Extract the flat vector and split it into one tensor per matrix, using
        the boundaries in ``weight_dimensions``, so the base contract receives
        a list of per-matrix segments.
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

        sizes = self._segment_sizes(self.weight_dimensions)
        total = sum(sizes)
        if vector.numel() != total:
            raise ValueError(
                f"Vector length {vector.numel()} does not match the sum of "
                f"weight_dimensions matrix sizes {total}."
            )

        segments: List[torch.Tensor] = []
        offset = 0
        for size in sizes:
            segments.append(vector[offset : offset + size])
            offset += size
        return segments

    # ------------------------------------------------------------------ #
    # slice
    # ------------------------------------------------------------------ #
    def slice_by_layers(
        self, flattened_weights: List[torch.Tensor]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Slice each per-matrix segment into fixed-size tokens. A matrix of
        ``n`` elements yields ``ceil(n / tokensize)`` tokens; only the final
        token of that matrix is right-padded. Each matrix gets its own index
        in ``position[:, 1]``.
        """
        outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for matrix_idx, w in enumerate(flattened_weights):
            if w.ndim != 1:
                raise ValueError(
                    f"Expected each segment to be 1D, got shape {tuple(w.shape)}"
                )

            n_values = w.numel()
            if n_values == 0:
                raise ValueError("Cannot tokenize an empty matrix segment.")

            remainder = n_values % self.tokensize
            pad_right = 0 if remainder == 0 else self.tokensize - remainder

            w_pad = self.pad(w, pad_right)
            n_tokens = w_pad.numel() // self.tokensize

            tokens = w_pad.view(n_tokens, self.tokensize)

            mask = torch.ones(
                (n_tokens, self.tokensize), dtype=torch.bool, device=tokens.device
            )
            if pad_right > 0:
                mask[-1, -pad_right:] = False

            position = torch.stack(
                [
                    torch.zeros(n_tokens, dtype=torch.long, device=tokens.device),
                    torch.full(
                        (n_tokens,), matrix_idx, dtype=torch.long, device=tokens.device
                    ),
                    torch.arange(n_tokens, dtype=torch.long, device=tokens.device),
                ],
                dim=1,
            )

            outputs.append((tokens, mask, position))

        return outputs

    # ------------------------------------------------------------------ #
    # unslice
    # ------------------------------------------------------------------ #
    def unslice(self, tokens_input, mask=None, position=None) -> List[torch.Tensor]:
        """
        Reconstruct per-matrix flattened segments from the tokenized
        representation.

        Supports:
        - full_model format: tokens_input, mask, position
        - layer_wise format: list of (tokens, mask, position)

        In full_model mode, tokens are grouped by matrix index, read from
        ``position[:, 1]`` (the layer-major stack written in slice_by_layers).
        Within each matrix, tokens are ordered by ``position[:, 2]``.
        """
        if isinstance(tokens_input, list):
            # layer-wise mode: list of tuples, already segmented by matrix
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

        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError(
                f"Expected position shape [num_tokens, 3], got {tuple(position.shape)}"
            )

        # Group by matrix index (position[:, 1]); sorted ascending.
        matrix_ids = torch.unique(position[:, 1], sorted=True)
        flattened_weights = []

        for matrix_id in matrix_ids:
            selector = position[:, 1] == matrix_id
            m_tokens = tokens_input[selector]
            m_mask = mask[selector]
            m_token_idx = position[selector, 2]

            # restore intra-matrix token order
            order = torch.argsort(m_token_idx)
            m_tokens = m_tokens[order]
            m_mask = m_mask[order]

            flat = m_tokens.reshape(-1)[m_mask.reshape(-1)]
            flattened_weights.append(flat)

        return flattened_weights

    # ------------------------------------------------------------------ #
    # rebuild
    # ------------------------------------------------------------------ #
    def rebuild_state_dict(
        self, flattened_weights: List[torch.Tensor], reference_statedict=None
    ) -> StateDict:
        """
        Rebuild a per-key LoRA state dict from per-matrix flattened segments,
        mirroring the reference ``unflatten``: each segment is reshaped to its
        ``real_shape`` (``weight_dimensions[key][1]``) and stored under its key.

        ``reference_statedict`` is accepted for signature compatibility but the
        boundaries/shapes come from ``self.weight_dimensions``.
        """
        dims = self._resolve_dims(self.weight_dimensions)
        keys = list(dims.keys())
        sizes = self._segment_sizes(self.weight_dimensions)
        real_shapes = self._segment_real_shapes(self.weight_dimensions)

        if len(flattened_weights) != len(keys):
            raise ValueError(
                f"Expected {len(keys)} segments (one per weight_dimensions "
                f"entry), got {len(flattened_weights)}."
            )

        final_weights: StateDict = {}
        for key, seg, size, real_shape in zip(
            keys, flattened_weights, sizes, real_shapes
        ):
            seg = seg.reshape(-1)
            if seg.numel() != size:
                raise ValueError(
                    f"Segment for '{key}' has {seg.numel()} elements but "
                    f"weight_dimensions expects {size}."
                )
            final_weights[key] = seg.unflatten(0, real_shape)

        return final_weights
import torch
import torch.nn as nn
import math

class LearnedPositionEmbeddings(nn.Module):
    r"""
    Adds learned positional embeddings to the inputs.

    This module generates learned positional embeddings for input tensors. The embeddings are learned during training and can handle 2D or 3D positional information.

    Args:
        embedding_dim (int): Dimension of the input embeddings.
        max_positions (list, optional): Maximum number of positions to embed for each dimension. Default: [1024, 256, 256]

    Attributes:
        max_positions (list): Maximum number of positions to embed for each dimension.
        embedding_dim (int): Dimension of the input embeddings.
        pe1 (nn.Embedding): Positional embedding layer for the first dimension.
        pe2 (nn.Embedding): Positional embedding layer for the second dimension.
        pe3 (nn.Embedding or None): Positional embedding layer for the third dimension, if applicable.

    Shape:
        - Input: `(B, N, embedding_dim)` where `B` is the batch size, `N` is the sequence length, and `embedding_dim` is the embedding dimension.
        - Output: `(B, N, embedding_dim)` where `B` is the batch size, `N` is the sequence length.

    """
    def __init__(self, embedding_dim: int, max_positions=[1024, 256, 256]):
        super().__init__()
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim

        assert len(max_positions) == 3, f"Expected max_positions to have 3 dimensions, but got max_positions={max_positions}."
        self.pe1 = nn.Embedding(max_positions[0], embedding_dim // 2)  # add 1 + 2
        self.pe2 = nn.Embedding(max_positions[1], embedding_dim // 2)  # add 1 + 2
        self.pe3 = nn.Embedding(max_positions[2], embedding_dim // 2)  # cat 1+2 & 3

    def forward(self, inputs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Applies the learned positional embeddings to the inputs.

        Args:
            inputs (torch.Tensor): Inputs to the layer, shape `(B, N, embedding_dim)`.
            pos (torch.Tensor): Position encodings for each token, shape `(B, N, 3)`.

        Returns:
            torch.Tensor: Output tensor with shape `(B, N, embedding_dim)`.

        Raises:
            AssertionError: If the number of dimensions of `inputs` is not 3.
            AssertionError: If the position tensors do not have as many dimensions as `max_positions`.
            AssertionError: If the position tensors do not have the same batch size as `inputs`.
            AssertionError: If the position tensors do not have the same sequence length as `inputs`.
        """
        assert (
            inputs.ndim == 3
        ), f"Number of dimensions should be 3, but input shape is {inputs.shape}"
        assert pos.shape[2] == len(
            self.max_positions
        ), f"Position tensors should have as many dimensions as max_positions ({len(self.max_positions)}), but got {pos.shape[2]}."
        assert (
            pos.shape[0] == inputs.shape[0]
        ), f"Position tensors ({pos.shape}) should have the same batch size as inputs ({inputs.shape})."
        assert (
            pos.shape[1] == inputs.shape[1]
        ), f"Position tensors ({pos.shape}) should have the same seq length as inputs ({inputs.shape})"

        pos_emb1 = self.pe1(pos[:, :, 0])
        pos_emb2 = self.pe2(pos[:, :, 1])

        if self.pe3 is not None:
            pos_emb3 = self.pe3(pos[:, :, 2])
            pos_emb = [pos_emb1 + pos_emb2, pos_emb3]
        else:
            pos_emb = [pos_emb1, pos_emb2]

        pos_emb = torch.cat(pos_emb, dim=2)
        out = inputs + pos_emb
        return out

class FunctionalSinusoidalPositionEmbeddings(nn.Module):
    r"""
    Applies sinusoidal positional embeddings with the assumption that the embedding dimension is even.

    This module generates sinusoidal positional embeddings dynamically for each positional dimension and sums them. The embeddings are added to the input tensor.

    Args:
        embedding_dim (int): Dimension of the embeddings.
        num_pos_dims (int, optional): Number of positional dimensions. Default: 3

    Attributes:
        embedding_dim (int): Dimension of the embeddings.
        num_pos_dims (int): Number of positional dimensions.

    Shape:
        - Input: `(B, N, embedding_dim)` or `(N, embedding_dim)` where `B` is the batch size, `N` is the sequence length, and `embedding_dim` is the embedding dimension.
        - Output: `(B, N, embedding_dim)` or `(N, embedding_dim)` where `B` is the batch size, `N` is the sequence length, and `embedding_dim` is the embedding dimension.

    """
    def __init__(self, embedding_dim: int, num_pos_dims: int = 3):
        super().__init__()
        assert embedding_dim % 2 == 0, "Embedding dimension must be even."
        
        self.embedding_dim = embedding_dim
        self.num_pos_dims = num_pos_dims
        
        # Precompute div_term once and register it as a buffer (so it's not a parameter but still moves with the model)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim))
        self.register_buffer("div_term", div_term)

    def _get_sinusoidal_embedding(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Generates sinusoidal embeddings dynamically for each positional dimension and sums them.

        Args:
            positions (torch.Tensor): Tensor of positions with shape `(B, N, num_pos_dims)`.

        Returns:
            torch.Tensor: Tensor of shape `(B, N, embedding_dim)` with the summed sinusoidal embeddings.
        """
        batch_size, seq_len, num_pos_dims = positions.shape

        # Create a tensor of shape (B, N, num_pos_dims, embedding_dim // 2)
        position_indices = positions.unsqueeze(-1).float() * self.div_term

        # Compute sin and cos parts
        sin_part = torch.sin(position_indices)
        cos_part = torch.cos(position_indices)

        # Initialize the embedding tensor
        emb = torch.zeros(batch_size, seq_len, self.embedding_dim, device=positions.device, dtype=torch.float32)

        # Use broadcasting to sum the sin and cos parts
        emb[:, :, 0::2] = sin_part.sum(dim=2)
        emb[:, :, 1::2] = cos_part.sum(dim=2)

        return emb

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Evaluates positional embeddings for inputs (handles both single sample and batch cases).

        Args:
            x (torch.Tensor): Inputs to the layer, shape `(B, N, embedding_dim)` or `(N, embedding_dim)`.
            pos (torch.Tensor): Position tensor of shape `(B, N, num_pos_dims)` or `(N, num_pos_dims)` for single samples.

        Returns:
            torch.Tensor: Output tensor with shape `(B, N, embedding_dim)` or `(N, embedding_dim)` for single samples.
        """
        single_sample = x.ndim == 2
        if single_sample:
            x = x.unsqueeze(0)
            pos = pos.unsqueeze(0)

        with torch.no_grad():  # Avoid tracking gradients for pos embeddings (saves memory in inference)
            pos_emb = self._get_sinusoidal_embedding(pos)

        x_with_pos = x + pos_emb

        return x_with_pos.squeeze(0) if single_sample else x_with_pos
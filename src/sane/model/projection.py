import torch
import torch.nn as nn
from typing import Tuple, Optional
from sane.utils.model_utils import get_activation_layer

class AverageProjectionHead(nn.Module):
    """
    Computes the embedding of the given hyper-representation by computing the average across tokens.

    This module takes an input tensor of shape :math:`(B, N, L)` and returns an output tensor of shape :math:`(B, L)`,
    where :math:`B` is the batch size, :math:`N` is the number of tokens, and :math:`L` is the latent dimension.

    Args:
        None

    Shape:
        - Input: :math:`(B, N, L)` where :math:`B` is the batch size, :math:`N` is the number of tokens, and :math:`L` is the latent dimension.
        - Output: :math:`(B, L)` where :math:`L` is the latent dimension.

    Examples:
        >>> average_projection_head = AverageProjectionHead()
        >>> input_tensor = torch.randn(32, 10, 20)  # Batch size of 32, 10 tokens, latent dimension of 20
        >>> output_tensor = average_projection_head(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([32, 20])
    """
    def __init__(self, **kwargs):
        super(AverageProjectionHead, self).__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AverageProjectionHead.

        Args:
            z (torch.Tensor): Input tensor of shape :math:`(B, N, L)`.

        Returns:
            torch.Tensor: Output tensor of shape :math:`(B, L)`.
        """
        return z.mean(dim=1)

class MLPProjectionHead(nn.Module):
    r"""
    Projects input tensors into a specified projection dimension using a multi-layer perceptron (MLP).

    This module takes an input tensor of shape :math:`(B, N, L)`, where :math:`B` is the batch size,
    :math:`N` is the number of tokens, and :math:`L` is the latent dimension. It projects the tensor
    into a tensor of shape :math:`(B, P)`, where :math:`P` is the projection dimension.

    Args:
        n_tokens (int): Number of tokens in the input tensor.
        latent_dim (int): Latent dimension of the input tensor.
        projection_dim (int, optional): Dimension to project the input tensor into. Default: 64.
        hidden_layers_dim (tuple, optional): Dimensions of the hidden layers. Default: (64,).
        activation (str, optional): Activation function to use. Default: 'relu'.

    Shape:
        - Input: :math:`(B, N, L)` where :math:`B` is the batch size, :math:`N` is the number of tokens, and :math:`L` is the latent dimension.
        - Output: :math:`(B, P)` where :math:`P` is the projection dimension.

    Attributes:
        n_tokens (int): Number of tokens in the input tensor.
        latent_dim (int): Latent dimension of the input tensor.
        projection_dim (int): Dimension to project the input tensor into.
        head (nn.Sequential): Sequential module containing the layers of the MLP.

    Examples:
        >>> projection_head = MLPProjectionHead(n_tokens=10, latent_dim=20, projection_dim=64)
        >>> input_tensor = torch.randn(32, 10, 20)  # Batch size of 32, 10 tokens, latent dimension of 20
        >>> output_tensor = projection_head(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([32, 64])
    """
    def __init__(self, n_tokens: int, latent_dim: int, projection_dim: int = 64, hidden_layers_dim: Tuple[int, ...] = (64,), activation: str = 'relu', **kwargs):
        super(MLPProjectionHead, self).__init__()
        assert isinstance(n_tokens, int) and (n_tokens > 0), "Number of tokens must be an integer greater than 0, but is {}".format(n_tokens)
        self.n_tokens = n_tokens
        assert isinstance(latent_dim, int) and (latent_dim > 0), "Latent dimension must be an integer greater than 0, but is {}".format(latent_dim)
        self.latent_dim = latent_dim
        assert isinstance(projection_dim, int) and (projection_dim > 0), "Projection dimension must be an integer greater than 0, but is {}".format(projection_dim)
        self.projection_dim = projection_dim

        layers = list()
        i_dim = self.n_tokens * self.latent_dim
        for hidden_dim in hidden_layers_dim:
            assert isinstance(hidden_dim, int) and (hidden_dim > 0), "Hidden layer dimension must be an integer greater than 0, but is {}".format(hidden_dim)
            layers.append(nn.Linear(i_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation_layer(activation))
            i_dim = hidden_dim
        layers.append(nn.Linear(i_dim, self.projection_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of the MLPProjectionHead.

        Args:
            z (torch.Tensor): Input tensor of shape :math:`(B, N, L)`.

        Returns:
            torch.Tensor: Output tensor of shape :math:`(B, P)`.
        """
        assert z.shape[-1] == self.latent_dim, f"MLPProjectionHead latent dim ({self.latent_dim}) does not match input tensor shape ({z.shape}) at index -1"
        assert z.shape[-2] == self.n_tokens, f"MLPProjectionHead n_tokens ({self.n_tokens}) does not match input tensor shape ({z.shape}) at index -2"
        z = z.view(z.shape[0], -1)
        return self.head(z)

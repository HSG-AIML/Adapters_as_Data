import torch
import torch.nn as nn

from typing import Optional, List

from sane.utils.model_utils import get_activation_layer


class ResidualMLPBlock(nn.Module):
    r"""
    Residual MLP block with pre-normalization.

    The block applies LayerNorm, a two-layer MLP with an internal expansion ratio,
    and adds a residual connection. When the input and output dimensions differ,
    a linear projection is applied to the residual path so the shapes match.

    The forward pass is::

        h = norm(x)
        h = activation(linear1(h))
        h = dropout(h)
        h = linear2(h)
        return h + residual_proj(x)

    Args:
        in_dim (int): Dimension of the input features.
        out_dim (int): Dimension of the output features.
        expansion (int, optional): Expansion ratio for the internal hidden dimension. Default: 2
        dropout (float, optional): Dropout probability. Default: 0.1
        activation (str, optional): Activation function name (see ``get_activation_layer``). Default: 'gelu'

    Attributes:
        in_dim (int): Dimension of the input features.
        out_dim (int): Dimension of the output features.
        norm (nn.LayerNorm): Pre-normalization layer applied to the input.
        linear1 (nn.Linear): First linear layer projecting to the expanded hidden dimension.
        activation (nn.Module): Activation function applied after ``linear1``.
        dropout (nn.Dropout): Dropout applied after the activation.
        linear2 (nn.Linear): Second linear layer projecting back to ``out_dim``.
        residual_proj (nn.Linear or nn.Identity): Projection of the residual path when shapes differ.

    Shape:
        - Input: `(B, in_dim)` where `B` is the batch size.
        - Output: `(B, out_dim)` where `B` is the batch size.
    """
    def __init__(self, in_dim: int, out_dim: int, expansion: int = 2, dropout: float = 0.1, activation: str = 'gelu'):
        super(ResidualMLPBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        hidden_dim = out_dim * expansion

        self.norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.activation = get_activation_layer(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualMLPBlock.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, in_dim)`.

        Returns:
            torch.Tensor: Output tensor of shape `(B, out_dim)`.
        """
        residual = self.residual_proj(x)
        h = self.norm(x)
        h = self.linear1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return h + residual


class ResidualMLP(nn.Module):
    r"""
    Stack of residual MLP blocks that progressively transforms the input dimension to the output dimension.

    The module builds a sequence of ``ResidualMLPBlock`` layers following the dimensions
    in ``hidden_dims``, and finishes with a linear projection to ``output_dim``. A final
    LayerNorm is applied before the output projection to stabilize training.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        hidden_dims (list): Sequence of hidden dimensions to traverse between input and output. Default: [4096, 1024, 256]
        expansion (int, optional): Expansion ratio inside each ``ResidualMLPBlock``. Default: 2
        dropout (float, optional): Dropout probability. Default: 0.1
        activation (str, optional): Activation function name. Default: 'gelu'

    Attributes:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        blocks (nn.ModuleList): The stacked residual MLP blocks.
        final_norm (nn.LayerNorm): Pre-output normalization layer.
        output_layer (nn.Linear): Final projection to ``output_dim``.

    Shape:
        - Input: `(B, input_dim)` where `B` is the batch size.
        - Output: `(B, output_dim)` where `B` is the batch size.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [4096, 1024, 256], expansion: int = 2, dropout: float = 0.1, activation: str = 'gelu'):
        super(ResidualMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        dims = [input_dim] + list(hidden_dims)
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(dims[i], dims[i + 1], expansion=expansion, dropout=dropout, activation=activation)
            for i in range(len(dims) - 1)
        ])

        self.final_norm = nn.LayerNorm(dims[-1])
        self.output_layer = nn.Linear(dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualMLP.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, input_dim)`.

        Returns:
            torch.Tensor: Output tensor of shape `(B, output_dim)`.
        """
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.output_layer(x)
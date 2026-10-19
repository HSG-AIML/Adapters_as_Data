import torch
import torch.nn as nn
from typing import Tuple, Optional
from sane.utils.model_utils import get_activation_layer

class MLPTranslator(nn.Module):
    """
    Translates the input tensor into a different latent space using a multi-layer perceptron (MLP).

    This module takes an input tensor of shape :math:`(B, input_n_tokens, input_dim)` and returns an output tensor of shape :math:`(B, output_n_tokens, output_dim)`,
    where :math:`B` is the batch size.

    Args:
        input_dim (int): Dimension of the input tensor.
        input_n_tokens (int): Number of tokens in the input tensor.
        output_dim (int, optional): Dimension of the output tensor. If None, it defaults to the input dimension. Default: None
        output_n_tokens (int, optional): Number of tokens in the output tensor. If None, it defaults to the input number of tokens. Default: None
        hidden_layers_dim (tuple, optional): Dimensions of the hidden layers. Default: (64,)
        activation (str, optional): Activation function to use. Default: 'relu'

    Shape:
        - Input: :math:`(B, input_n_tokens, input_dim)` where :math:`B` is the batch size.
        - Output: :math:`(B, output_n_tokens, output_dim)`
    """

    def __init__(self, input_dim: int, input_n_tokens: int, output_dim: Optional[int] = None, output_n_tokens: Optional[int] = None, hidden_layers_dim: Tuple[int] = (64,), activation: str = 'relu', **kwargs):
        super(MLPTranslator, self).__init__()
        self.input_dim = input_dim
        self.input_n_tokens = input_n_tokens
        self.output_dim = input_dim if output_dim is None else output_dim
        self.output_n_tokens = input_n_tokens if output_n_tokens is None else output_n_tokens
        
        assert isinstance(self.input_dim, int) and (self.input_dim > 0), "Input dimension must be an integer greater than 0, but is {}".format(input_dim)
        assert isinstance(self.input_n_tokens, int) and (self.input_n_tokens > 0), "Input number of tokens must be an integer greater than 0, but is {}".format(input_n_tokens)
        assert isinstance(self.output_dim, int) and (self.output_dim > 0), "Output dimension must be an integer greater than 0, but is {}".format(self.output_dim)
        assert isinstance(self.output_n_tokens, int) and (self.output_n_tokens > 0), "Output number of tokens must be an integer greater than 0, but is {}".format(self.output_n_tokens)

        layers = list()
        i_dim = self.input_dim * self.input_n_tokens
        for hidden_dim in hidden_layers_dim:
            layers.append(nn.Linear(i_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation_layer(activation))
            i_dim = hidden_dim
        layers.append(nn.Linear(i_dim, self.output_dim * self.output_n_tokens))
        self.head = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLPTranslator.

        Args:
            z (torch.Tensor): Input tensor of shape :math:`(B, input_n_tokens, input_dim)`.

        Returns:
            torch.Tensor: Output tensor of shape :math:`(B, output_n_tokens, output_dim)`.
        """
        z = z.view(z.shape[0], -1)  # Flatten the input tensor
        z = self.head(z)
        return z.view(z.shape[0], self.output_n_tokens, self.output_dim)

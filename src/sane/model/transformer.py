import torch
import torch.nn as nn

from typing import Optional

from sane.model.position import LearnedPositionEmbeddings, FunctionalSinusoidalPositionEmbeddings
from sane.model.gpt_transformer import GPTTransformerEncoder

class SANETransformerEncoder(nn.Module):
    r"""
    Transformer Encoder module that processes input tensors through a series of transformer layers.

    This module first projects the input to the desired model dimension using a linear layer,
    then adds positional embeddings to the input. The resulting tensor is passed through a
    series of transformer encoder layers, and finally projected back to the output dimension.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        d_model (int): Dimension of the model.
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer encoder layers.
        dropout (float, optional): Dropout probability. Default: 0.1
        position_embedding (str, optional): Type of positional embedding to use ('learned' or 'sinusoidal'). Default: 'learned'
        max_positions (list, optional): Maximum positions for each dimension of the positional embeddings, which is used for the learned embeddings. Default: [1024, 256, 256]
        num_pos_dims (int, optional): Number of dimensions in the position vector, which is used for the sinusoidal embeddings. Default: 3

    Attributes:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        d_model (int): Dimension of the model.
        input_layer (nn.Linear or nn.Identity): Layer to project input to model dimension.
        position_embedding (LearnedPositionEmbeddings or FunctionalSinusoidalPositionEmbeddings): Positional embedding module.
        transformer_encoder (nn.TransformerEncoder): Transformer encoder layers.
        output_layer (nn.Linear or nn.Identity): Layer to project output back to output dimension.

    Shape:
        - Input: `(B, N, input_dim)` where `B` is the batch size, `N` is the number of tokens.
        - Output: `(B, N, output_dim)` where `B` is the batch size, `N` is the number of tokens.

    """
    def __init__(self, input_dim: int, output_dim: int, d_model: int, nhead: int, num_layers: int, n_tokens: Optional[int] = None, transformer_type: str = 'pytorch', dropout: float = 0.1, position_embedding: str = 'learned', max_positions: Optional[list] = [1024, 256, 256], num_pos_dims: int = 3, block_size: Optional[int] = None, **kwargs):
        super(SANETransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.n_layers = num_layers
        
        # Use bias=False for GPT transformer type (matches GPT-2)
        use_bias = transformer_type != 'gpt'
        
        self.input_layer = nn.Linear(input_dim, d_model, bias=use_bias) if input_dim != d_model else nn.Identity()

        if position_embedding == 'learned':
            if max_positions is None:
                raise ValueError("max_positions must be specified for learned position embeddings.")
            self.position_embedding = LearnedPositionEmbeddings(d_model, max_positions)
        elif position_embedding == 'sinusoidal':
            self.position_embedding = FunctionalSinusoidalPositionEmbeddings(d_model, num_pos_dims=num_pos_dims)
        else:
            raise ValueError(f"Unknown position embedding type: {position_embedding}")

        if transformer_type == 'pytorch':
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=num_layers
            )
        elif transformer_type == 'gpt':
            # Use explicit block_size if provided, otherwise fall back to n_tokens (window size)
            effective_block_size = block_size if block_size is not None else n_tokens
            if effective_block_size is None:
                raise ValueError("block_size or n_tokens must be specified for GPT transformer.")

            self.transformer_encoder = GPTTransformerEncoder(
                n_layer=num_layers,
                d_model=d_model,
                n_head=nhead,
                dropout=dropout,
                bias=False,
                causal=False,
                block_size=effective_block_size,
            )
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}. Supported types are 'pytorch' and 'gpt'.")
        
        self.output_layer = nn.Linear(d_model, output_dim, bias=use_bias) if d_model != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SANETransformerEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, N, input_dim)`, where `B` is the batch size, `N` is the number of tokens, and `L` is the input dimension.
            pos (torch.Tensor): Position tensor of shape `(B, N, 3)` for learned or sinusoidal embeddings.
            src_key_padding_mask (torch.Tensor, optional): Optional boolean mask tensor for padding entire tokens, of shape (B, N). Batch elements and tokens that are to be ignored should have `True` in the mask. Please note that this mask is different from the one generated by the tokenizer.

        Returns:
            torch.Tensor: Output tensor of shape `(B, N, output_dim)`.
        """
        x = self.input_layer(x)
        x = self.position_embedding(x, pos)
        x = self.transformer_encoder(x)
        return self.output_layer(x)

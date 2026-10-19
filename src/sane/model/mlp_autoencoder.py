import torch
import torch.nn as nn

from typing import Optional, Dict, List

from sane.model.mlp import ResidualMLP
from sane.utils.model_utils import initialize_weights


class SANEMLPAutoEncoder(nn.Module):
    r"""
    Feedforward autoencoder for SANE that encodes a single flat input vector (e.g. a LoRA adapter)
    into a low-dimensional latent vector and reconstructs the input from it.

    Unlike :class:`SANEAutoEncoder`, this model does not use positional encodings, tokens, or attention.
    It is intended for the regime where each sample is presented as a single high-dimensional vector
    rather than a sequence of tokens. The forward signature accepts ``p`` and ``mask`` for compatibility
    with training pipelines built around the Transformer-based autoencoder, but ignores them.

    The encoder is a stack of residual MLP blocks that progressively reduces the dimension to
    ``latent_dim``. The decoder mirrors the encoder by default, mapping the latent vector back to
    the input dimension.

    Args:
        encoder (nn.Module, optional): The encoder module. Defaults to a ``ResidualMLP`` mapping 99648 to 64 via hidden dims [4096, 1024, 256].
        decoder (nn.Module, optional): The decoder module. Defaults to a ``ResidualMLP`` mapping 64 to 99648 via hidden dims [256, 1024, 4096].

    Attributes:
        encoder (nn.Module): Module mapping the input vector to the latent vector.
        decoder (nn.Module): Module mapping the latent vector back to the input space.
    """

    def __init__(self,
                 encoder: nn.Module = ResidualMLP(input_dim=99648, output_dim=64, hidden_dims=[4096, 1024, 256]),
                 decoder: nn.Module = ResidualMLP(input_dim=64, output_dim=99648, hidden_dims=[256, 1024, 4096])):
        super(SANEMLPAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        initialize_weights(self)

    @classmethod
    def from_config(cls, config: dict) -> 'SANEMLPAutoEncoder':
        r"""
        Constructs a SANEMLPAutoEncoder instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the encoder and decoder.

        Returns:
            SANEMLPAutoEncoder: An instance of the SANEMLPAutoEncoder class configured according to the provided dictionary.

        Examples::
            >>> config = {
            ...     'input_dim': 99648,
            ...     'latent_dim': 64,
            ...     'encoder': {
            ...         'hidden_dims': [4096, 1024, 256],
            ...         'expansion': 2,
            ...         'dropout': 0.1,
            ...         'activation': 'gelu',
            ...     },
            ...     'decoder': 'symmetric',
            ...     'device': 'cuda'
            ... }
            >>> model = SANEMLPAutoEncoder.from_config(config)
        """
        # Get the dimensions from the config
        input_dim = config.get('input_dim', 99648)
        latent_dim = config.get('latent_dim', 64)

        # Set up the encoder
        encoder_cfg = dict(config['encoder'])
        encoder_cfg.pop('architecture', None)
        encoder = ResidualMLP(input_dim=input_dim, output_dim=latent_dim, **encoder_cfg)

        # Set up the decoder
        if config['decoder'] == 'symmetric':
            decoder_cfg = dict(config['encoder'])
            decoder_cfg.pop('architecture', None)
            if 'hidden_dims' in decoder_cfg:
                decoder_cfg['hidden_dims'] = list(reversed(decoder_cfg['hidden_dims']))
        else:
            decoder_cfg = dict(config['decoder'])
            decoder_cfg.pop('architecture', None)

        decoder_output_dim = decoder_cfg.pop('output_dim', input_dim)
        decoder = ResidualMLP(input_dim=latent_dim, output_dim=decoder_output_dim, **decoder_cfg)

        # Set the device
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        return cls(encoder=encoder, decoder=decoder).to(device=device)

    def forward(self, x: torch.Tensor, p: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        r"""
        Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input tensor of shape `(B, input_dim)`.
            p (Optional[torch.Tensor]): Accepted for API compatibility with :class:`SANEAutoEncoder` and ignored.
            mask (Optional[torch.Tensor]): Accepted for API compatibility with :class:`SANEAutoEncoder` and ignored.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the input tensor ``x``, the latent tensor ``z``, and the reconstructed tensor ``x_hat``.
        """
        out = {'x': x}
        if p is not None:
            out['p'] = p
        if mask is not None:
            out['mask'] = mask

        out['z'] = self.encoder(x)
        out['x_hat'] = self.decoder(out['z'])
        return out

    def forward_embeddings(self, x: torch.Tensor, p: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
        Computes the latent embedding of the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape `(B, input_dim)`.
            p (Optional[torch.Tensor]): Accepted for API compatibility and ignored.

        Returns:
            torch.Tensor: The latent embedding of shape `(B, latent_dim)`.
        """
        return self.encoder(x)
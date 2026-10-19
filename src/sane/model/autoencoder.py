import torch
import torch.nn as nn
from typing import Optional, Dict

from sane.model.transformer import SANETransformerEncoder
from sane.model.projection import AverageProjectionHead, MLPProjectionHead
from sane.model.translator import MLPTranslator
from sane.utils.model_utils import initialize_weights, gpt2_init_weights

class SANEAutoEncoder(nn.Module):
    r"""
    Skeleton class for the SANE AutoEncoder which is modular and can be configured with different modules.
    Please refer to the README.md for more details on the architecture.

    Args:
        encoder (nn.Module, optional): The encoder module. Defaults to a SANETransformerEncoder with input_dim=256, output_dim=128, d_model=512, nhead=8, num_layers=4.
        decoder (nn.Module, optional): The decoder module. Defaults to a SANETransformerEncoder with input_dim=128, output_dim=256, d_model=512, nhead=8, num_layers=4.
        embedder (nn.Module, optional): The embedder module. Defaults to an AverageProjectionHead.
        projector (nn.Module, optional): The projector module. Defaults to None.
        translator (nn.Module, optional): The translator module. Defaults to None.
    """

    def __init__(self,
                 encoder: nn.Module = SANETransformerEncoder(input_dim=256, output_dim=128, d_model=512, nhead=8, num_layers=4),
                 decoder: nn.Module = SANETransformerEncoder(input_dim=128, output_dim=256, d_model=512, nhead=8, num_layers=4),
                 embedder: nn.Module = AverageProjectionHead(),
                 projector: Optional[nn.Module] = None,
                 translator: Optional[nn.Module] = None,
                 use_gpt_init: bool = False,
                 n_layer: Optional[int] = None):
        super(SANEAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedder = embedder
        self.projector = projector
        self.translator = translator

        # Use GPT-2 style initialization for GPT transformers (encoder/decoder only),
        # otherwise use standard initialization for the whole model
        if use_gpt_init:
            gpt2_init_weights(self.encoder, n_layer=n_layer)
            gpt2_init_weights(self.decoder, n_layer=n_layer)
            # embedder, projector, translator keep PyTorch default init
        else:
            initialize_weights(self)

    @classmethod
    def from_config(cls, config: dict) -> 'SANEAutoEncoder':
        r"""
        Constructs a SANEAutoEncoder instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the encoder, decoder, embedder, projector, and translator.

        Returns:
            SANEAutoEncoder: An instance of the SANEAutoEncoder class configured according to the provided dictionary.

        Examples::
            >>> config = {
            ...     'input_dim': 256,
            ...     'latent_dim': 128,
            ...     'n_tokens': 16,
            ...     'encoder': {
            ...         'architecture': 'transformer',
            ...         'd_model': 512,
            ...         'nhead': 8,
            ...         'num_layers': 4,
            ...         'max_positions': [50000, 256, 1024]
            ...     },
            ...     'decoder': {
            ...         'architecture': 'transformer',
            ...         'd_model': 512,
            ...         'nhead': 8,
            ...         'num_layers': 4,
            ...         'max_positions': [50000, 256, 1024],
            ...     },
            ...     'embedder': {
            ...         'architecture': 'average',
            ...     },
            ...     'projector': {
            ...         'architecture': 'average',
            ...     },
            ...     'device': 'cuda'
            ... }
            >>> model = SANEAutoEncoder.from_config(config)
        """
        # Get the dimensions from the config
        input_dim = config.get('input_dim', 256)
        latent_dim = config.get('latent_dim', 128)
        n_tokens = config.get('n_tokens', None)

        # Set up the encoder
        if config['encoder'].get('architecture', 'transformer') == 'transformer':
            encoder = SANETransformerEncoder(input_dim=input_dim, output_dim=latent_dim, n_tokens=n_tokens, max_positions=config.get('max_positions', None), **config['encoder'])
        else:
            raise ValueError(f"Unknown encoder architecture: {config['encoder']['architecture']}")
        
        # Set up the translator
        if 'translator' not in config or config['translator'] is None:
            translator = None
        elif config['translator']['architecture'] == 'mlp':
            assert n_tokens is not None, "`n_tokens` must be specified in the config if you intend to use MLPTranslator."
            translator = MLPTranslator(input_dim=latent_dim, input_n_tokens=n_tokens, **config['translator'])
        else:
            raise ValueError(f"Unknown translator architecture: {config['translator']['architecture']}")

        # Set up the decoder
        decoder_input_dim = latent_dim if translator is None else translator.output_dim
        decoder_n_tokens = n_tokens if translator is None else translator.output_n_tokens
        if config['decoder'] == 'symmetric':
            config['decoder'] = config['encoder'].copy()
        if 'output_dim' not in config['decoder']:
            config['decoder']['output_dim'] = input_dim
        if config['decoder'].get('architecture', 'transformer') == 'transformer':
            decoder = SANETransformerEncoder(input_dim=decoder_input_dim, n_tokens=decoder_n_tokens, max_positions=config.get('max_positions', None), **config['decoder'])
        else:
            raise ValueError(f"Unknown decoder architecture: {config['decoder']['architecture']}")

        # Set up the embedder
        if config['embedder'].get('architecture', 'average') == 'average':
            embedder = AverageProjectionHead(**config['embedder'])
        elif config['embedder'].get('architecture', 'average') == 'mlp':
            assert n_tokens is not None, "`n_tokens` must be specified in the config if you intend to use MLPProjectionHead."
            embedder = MLPProjectionHead(n_tokens=n_tokens, latent_dim=latent_dim, **config['embedder'])
        else:
            raise ValueError(f"Unknown embedder architecture: {config['embedder']['architecture']}")

        # Set up the projector
        if 'projector' not in config or config['projector'] is None:
            projector = None
        elif config['projector'].get('architecture', 'average') == 'average':
            projector = AverageProjectionHead(**config['projector'])
        elif config['projector'].get('architecture', 'average') == 'mlp':
            assert n_tokens is not None, "`n_tokens` must be specified in the config if you intend to use MLPProjectionHead."
            projector = MLPProjectionHead(n_tokens=n_tokens, latent_dim=latent_dim, **config['projector'])
        else:
            raise ValueError(f"Unknown projector architecture: {config['projector']['architecture']}")

        # Determine if GPT initialization should be used
        use_gpt_init = config['encoder'].get('transformer_type', 'pytorch') == 'gpt'
        n_layer = config['encoder'].get('num_layers', None) if use_gpt_init else None

        # Set the device
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        return cls(encoder=encoder, decoder=decoder, embedder=embedder, projector=projector, translator=translator, use_gpt_init=use_gpt_init, n_layer=n_layer).to(device=device)

    def forward(self, x: torch.Tensor, p: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        r"""
        Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input tensor.
            p (torch.Tensor): The positional encoding tensor.
            mask (Optional[torch.Tensor]): An optional mask tensor to apply to the input. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the output tensors 'z', 'z_tilde' (if translator is not None), 'x_hat', and 'z_p' (if projector is not None).
        """
        out = {
            'x': x,
            'p': p
        }
        if mask is not None:
            out['mask'] = mask

        # Encoder 
        out['z'] = self.encoder(x, p)

        # Projector
        if self.projector is not None:
            out['z_p'] = self.projector(out['z'])

        # Translator & Decoder
        if self.translator is not None:
            out['z_tilde'] = self.translator(out['z'])

            # Temporary implementation: generate synthetic positional encoding for the translator output if necessary
            # These synthetic positions simulate a 1-layer NN that includes all tokens
            # Therefore, the first and third columns of the position tensor are torch.arange(out['z_tilde'].shape[1]) while the second column is all zeros.
            if out['z_tilde'].shape[1] != p.shape[1]:
                p = torch.stack((
                    torch.arange(out['z_tilde'].shape[1]),
                    torch.zeros(out['z_tilde'].shape[1]),
                    torch.arange(out['z_tilde'].shape[1])
                ), dim=1).unsqueeze(0).repeat(p.shape[0], 1, 1).to(device=p.device, dtype=p.dtype)

                assert p.shape == torch.Size([out['z_tilde'].shape[0], out['z_tilde'].shape[1], 3]), f"Expected shape {(out['z_tilde'].shape[0], out['z_tilde'].shape[1], 3)}, but got {p.shape}."

            out['x_hat'] = self.decoder(out['z_tilde'], p)
        else:
            out['x_hat'] = self.decoder(out['z'], p)
        return out

    def forward_embeddings(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the embeddings of the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            p (torch.Tensor): The positional encoding tensor.

        Returns:
            torch.Tensor: The embeddings of the input tensor.
        """
        z = self.encoder(x, p)
        return self.embedder(z)

    def forward_projector(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the projected embeddings of the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            p (torch.Tensor): The positional encoding tensor.

        Returns:
            torch.Tensor: The projected embeddings of the input tensor.
        """
        z = self.encoder(x, p)
        return self.projector(z)
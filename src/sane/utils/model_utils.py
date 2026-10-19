import torch
import torch.nn as nn
import math 

def get_activation_layer(activation: str) -> nn.Module:
    """Returns the activation layer based on the specified activation function."""
    activation_map = {
        'relu': nn.ReLU,
        'leakyrelu': nn.LeakyReLU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'prelu': nn.PReLU,
        'selu': nn.SELU,
        'silu': nn.SiLU,
        'celu': nn.CELU,
        'gelu': nn.GELU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
    }
    return activation_map[activation]()

def initialize_weights(module: nn.Module, initialization: str = 'xavier_uniform') -> None:
    """
    Initializes the weights of a model based on the specified initialization method.

    Args:
        module: The PyTorch module whose weights are to be initialized.
        initialization: The initialization method to use. Options are:
            - 'xavier_uniform': Xavier uniform initialization
            - 'xavier_normal': Xavier normal initialization
            - 'kaiming_uniform': Kaiming uniform initialization
            - 'kaiming_normal': Kaiming normal initialization
            - 'uniform': Uniform initialization
            - 'normal': Normal initialization
    """
    if not isinstance(module, nn.Module):
        raise TypeError("The provided module must be an instance of nn.Module")
    elif isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        # Initialize weights for linear and convolutional layers
        match initialization:
            case 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            case 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            case 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            case 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            case 'uniform':
                nn.init.uniform_(module.weight)
            case 'normal':
                nn.init.normal_(module.weight)
            case _:
                raise ValueError(f"Unknown initialization method: {initialization}")
        
        # Initialize bias to a small constant if it exists
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.01)
        
    for child in module.children():
        # Recursively initialize weights for child modules
        initialize_weights(child, initialization)
        
def gpt2_init_weights(module: nn.Module, n_layer: int = None) -> None:
    """
    GPT-2 style initialization (from Karpathy's nanoGPT / SHRP).
    
    Args:
        module: The PyTorch module to initialize
        n_layer: Number of layers for scaled residual projection init
    """
    def _init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    module.apply(_init)
    
    # Scale residual projections
    if n_layer is not None:
        for pn, p in module.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

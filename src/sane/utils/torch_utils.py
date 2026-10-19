import torch

def batch_info(batch, indent=0):
    """Returns a formatted string describing the structure and shapes of a batch.

    Recursively traverses a batch, which can be a list, tuple, dict, or tensor,
    and returns a string describing its structure and the shapes of all tensors.

    Args:
        batch: The batch to describe. Can be a list, tuple, or torch.Tensor.
        indent (int, optional): Current indentation level for pretty formatting.
            Defaults to 0.

    Returns:
        str: A formatted string describing the batch structure and tensor shapes.
    """
    indent_str = "  " * indent
    result = []
    if isinstance(batch, list):
        result.append(f"{indent_str}List of length: {len(batch)}")
        for i, item in enumerate(batch):
            result.append(f"{indent_str}  Item {i}:")
            result.append(batch_info(item, indent + 2))
    elif isinstance(batch, tuple):
        result.append(f"{indent_str}Tuple of length: {len(batch)}")
        for i, item in enumerate(batch):
            result.append(f"{indent_str}  Element {i}:")
            result.append(batch_info(item, indent + 2))
    elif isinstance(batch, torch.Tensor):
        result.append(f"{indent_str}Tensor with shape: {batch.shape}")
    elif isinstance(batch, dict):
        result.append(f"{indent_str}Dictionary of length: {len(batch)}")
        for k, v in batch.items():
            result.append(f"{indent_str}  Key {k}:")
            result.append(batch_info(v, indent + 2))
    elif isinstance(batch, (int, float, bool, str)):
        result.append(f"{indent_str}{type(batch).__name__}: {batch}")
    else:
        result.append(f"{indent_str}Unknown type: {type(batch)}")
    return "\n".join(result)

def model_size_info(model):
    """Estimates the memory size of a PyTorch model.

    Args:
        model (torch.nn.Module): The model to estimate.

    Returns:
        dict: A dictionary containing information about the model's size and memory usage.
    """
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    size += sum(p.numel() * p.element_size() for p in model.buffers())
    return {
        "class": model.__class__.__name__,
        "device": next(iter(model.parameters())).device,
        "dtype": next(iter(model.parameters())).dtype,
        "num_weights": sum(p.numel() for p in model.parameters()),
        "num_buffers": sum(b.numel() for b in model.buffers()),
        "size_mb": size / (1024 ** 2),  # Convert bytes to megabytes
        "size_gb": "{:.2f} GB".format(size / (1024 ** 3))
    }

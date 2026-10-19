import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from math import isclose

class SANELoss(nn.Module, ABC):
    """
    Base class for SANE loss functions.

    This class defines the interface for all SANE loss functions. It inherits from nn.Module and requires subclasses to implement the `forward` method.

    Args:
        name (str): The name of the loss function. If the name does not start with 'loss_', it will be prepended.

    Returns:
        None
    """
    def __init__(self, name: str) -> None:
        super(SANELoss, self).__init__()
        if not isinstance(name, str):
            raise TypeError(f"Expected name to be a string, got {type(name)}")
        if not name.startswith("loss_"):
            name = "loss_" + name
        self.name: str = name

    @abstractmethod
    def forward(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the loss given a dictionary of tensors.

        Args:
            tensors (Dict[str, torch.Tensor]): A dictionary containing the input tensors required for the loss computation.
                The keys are strings that identify the tensors, and the values are the corresponding torch.Tensor objects.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the computed loss tensors.
                The key `loss` should be present and represents the main loss component.
                The key corresponding to the loss function's name will also be present and should contain the same value as `loss`.
                It may contain additional keys for other loss components.
        """
        pass

class CompositeLoss(SANELoss):
    """
    A composite loss function that combines multiple SANE loss functions.

    This class allows for the combination of multiple SANE loss functions into a single loss function.
    It takes a list of SANE loss functions and computes their weighted sum based on the provided ratios.

    Args:
        name (str): The name of the composite loss function.
        losses (List[SANELoss]): A list of SANE loss functions to combine.
        ratios (List[float]): A list of ratios for each loss function. The ratios determine the weight of each loss function in the final composite loss.
            If the sum of the ratios is less than or equal to 1 and there is one more loss than ratios, the last loss is treated as a residual loss with ratio `1.0 - sum(ratios)`.

    Raises:
        ValueError: If the number of losses does not match the number of ratios, if any ratio is negative or if the sum of ratios is not greater than 0.
        TypeError: If any ratio is not a float or int, or if any ratio is negative.

    Returns:
        None
    """
    def __init__(self, name: str, losses: List[SANELoss], ratios: List[float]) -> None:
        super(CompositeLoss, self).__init__(name)
        for ratio in ratios:
            if not isinstance(ratio, (float, int)):
                raise TypeError(f"Expected ratio to be a float or int, got {type(ratio)}")
            if ratio < 0:
                raise ValueError(f"Ratios must be non-negative, got {ratio}")
        if len(losses) - len(ratios) == 1 and sum(ratios) <= 1:
            # If there is one more loss than ratios, we assume the last loss is the residual loss
            ratios = ratios + [1.0 - sum(ratios)]
        if len(losses) != len(ratios):
            raise ValueError("The number of losses must match the number of ratios.")
        if sum(ratios) <= 0:
            raise ValueError(f"The sum of ratios must be greater than zero but is {sum(ratios)} ({ratios}).")
        self.losses: nn.ModuleList = nn.ModuleList(losses)
        self.ratios: List[float] = ratios

    def forward(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the composite loss given a dictionary of tensors.

        Args:
            tensors (Dict[str, torch.Tensor]): A dictionary containing the input tensors required for the loss computation.
                The keys are strings that identify the tensors, and the values are the corresponding torch.Tensor objects.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the computed loss tensors.
                The key `loss` represents the main composite loss component, which is the weighted sum of the individual loss components.
                It may contain additional keys for other loss components from the individual loss functions.
        """
        device = next(iter(tensors.values())).device
        output: Dict[str, torch.Tensor] = {'loss': torch.zeros(1, device=device)}
        for loss, ratio in zip(self.losses, self.ratios):
            if not isclose(ratio, 0.):
                loss_output = loss(tensors)
                output['loss'] += loss_output.pop('loss') * ratio
                output.update(loss_output)
            else:
                # If the ratio is zero, we set the corresponding losses to zero
                for subloss in loss.modules():
                    if isinstance(subloss, SANELoss):
                        output[subloss.name] = torch.zeros(1, device=device)
        output[self.name] = output['loss']
        return output

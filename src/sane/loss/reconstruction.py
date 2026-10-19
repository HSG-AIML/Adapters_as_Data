import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from torchmetrics.functional import r2_score
from sane.loss.loss import SANELoss

class MSEReconstructionLoss(SANELoss):
    """
    Mean Squared Error (MSE) Reconstruction Loss.

    This loss function computes the masked MSE between the input and the target, eventually using masked per-token loss normalization.

    Args:
        name (str, optional): The name of the loss. Default: "loss_mse_reconstruction"
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'
        normalize (str, optional): What normalization strategy to use. Can be None, 'token', 'window' or 'batch'. Default: 'token'
        std_eps (float, optional): A small value added to the standard deviation to avoid division by zero. Default: 1e-6

    Shape:
        - Input: Dict[str, torch.Tensor] containing 'x', 'x_hat', and 'mask' tensors.
        - Output: Dict[str, torch.Tensor] containing the loss and R-squared score.

    Examples:
        >>> loss_fn = MSEReconstructionLoss()
        >>> tensors = {'x': torch.randn(3, 5), 'x_hat': torch.randn(3, 5), 'mask': torch.ones(3, 5)}
        >>> loss = loss_fn(tensors)
    """

    def __init__(self, name: str = "loss_mse_reconstruction", reduction: str = "mean", normalize: str = 'token', std_eps: float = 1e-6) -> None:
        super(MSEReconstructionLoss, self).__init__(name)
        self.reduction = reduction
        self.normalize = normalize
        self.std_eps = std_eps

    def forward(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute the MSE loss.
        If the input includes multiple views, the loss is computed for each view and averaged.

        Args:
            tensors (Dict[str, torch.Tensor]): A dictionary containing the following tensors:
                - 'x': The target tensor.
                - 'x_hat': The output tensor.
                - 'mask': The mask tensor. If not provided, a tensor of ones will be used.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the loss and R-squared score.
        """
        # If there are multiple view, compute the loss for each view
        if 'x_i' in tensors and 'x' not in tensors:
            loss_i = self.forward({
                'x': tensors['x_i'],
                'x_hat': tensors['x_hat_i'],
                'mask': tensors['mask_i'],
                'z': tensors['z_i']
            })
            loss_j = self.forward({
                'x': tensors['x_j'],
                'x_hat': tensors['x_hat_j'],
                'mask': tensors['mask_j'],
                'z': tensors['z_j']
            })
            return {k: (loss_i[k] + loss_j[k]) / 2 for k in loss_i.keys()}

        # Compute the MSE loss
        target = tensors['x']
        output = tensors['x_hat']
        mask = tensors['mask'].to(dtype=torch.bool) if 'mask' in tensors else torch.ones_like(target, dtype=torch.bool, device=target.device)

        if torch.isnan(target).any():
            raise ValueError(f"Target tensor contains {torch.isnan(target).sum().item()}/{target.numel()} NaN values")
        if torch.isnan(output).any():
            raise ValueError(f"Output tensor contains {torch.isnan(output).sum().item()}/{output.numel()} NaN values")
        if torch.isnan(mask).any():
            raise ValueError(f"Mask tensor contains {torch.isnan(mask).sum().item()}/{mask.numel()} NaN values")

        assert (
            output.shape == target.shape == mask.shape
        ), f"MSE loss error: prediction and target don't have the same shape. output {output.shape} vs target {target.shape} vs mask {mask.shape}"

        # We eventually apply the masked per-token loss normalization
        if self.normalize is None or not self.normalize:
            target_normalized = target
            output_normalized = output
        elif self.normalize == "batch":
            # flatten across batch, window, token
            selected = torch.masked_select(target, mask)
            target_mean = selected.mean()
            target_std = selected.std() + self.std_eps
            target_normalized = (target - target_mean) / target_std
            output_normalized = (output - target_mean) / target_std
        elif self.normalize == "window":
            # standardize per sequence (window)
            mask_f = mask.float()
            sum_mask = mask_f.sum(dim=(1,2), keepdim=True).clamp(min=1)
            target_mean = (target.sum(dim=(1,2), keepdim=True) / sum_mask)
            target_var  = ((target - target_mean)**2 * mask_f).sum(dim=(1,2), keepdim=True) / sum_mask
            target_std = torch.sqrt(target_var) + self.std_eps
            target_normalized = (target - target_mean) / target_std
            output_normalized = (output - target_mean) / target_std
        elif self.normalize == "token":
            # standardize per token dim across batch and window
            mask_f = mask.float()
            sum_mask = mask_f.sum(dim=(0,1), keepdim=True).clamp(min=1)
            target_mean = target.sum(dim=(0,1), keepdim=True) / sum_mask
            target_var  = ((target - target_mean)**2 * mask_f).sum(dim=(0,1), keepdim=True) / sum_mask
            target_std = torch.sqrt(target_var) + self.std_eps
            target_normalized = (target - target_mean) / target_std
            output_normalized = (output - target_mean) / target_std
        else:
            raise ValueError(f"Invalid normalization {self.normalize}: should be one of [None, 'batch', 'window', 'token']")

        target = torch.masked_select(target, mask)
        output = torch.masked_select(output, mask)

        target_normalized = torch.masked_select(target_normalized, mask)
        output_normalized = torch.masked_select(output_normalized, mask)

        loss = nn.functional.mse_loss(output_normalized, target_normalized, reduction=self.reduction)
        rsq = r2_score(output, target, multioutput="uniform_average")

        mask_ratio = mask.sum() / mask.numel()
        compression_ratio = mask.sum() / tensors['z'].numel() if 'z' in tensors else None

        return {
            'loss': loss,
            self.name: loss,
            self.name + '_rsq': rsq,
            'mask_ratio': mask_ratio,
            'compression_ratio': compression_ratio
        }
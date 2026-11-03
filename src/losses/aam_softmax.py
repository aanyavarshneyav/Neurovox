"""Additive Angular Margin Softmax (AAMSoftmax / ArcFace) loss module.

Implements Eq. (3) from Alam et al. (IJCNN 2025) with corrected target summation index (j != y_i, ADR-001):
    L = - (1/N) * sum_i log( exp(s * cos(theta_{y_i} + m)) /
                             ( exp(s * cos(theta_{y_i} + m)) + sum_{j != y_i} exp(s * cos(theta_j)) ) )

Features:
- Fixed fp32 execution for numerical stability (ADR-007).
- Monotonic margin fallback when theta + m > pi (th / mm overflow guard).
- Fully differentiable class prototype weight matrix.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax loss module.
    
    Args:
        in_features: Dimension of input speaker embeddings (e.g. 192).
        num_classes: Number of training speaker classes (e.g. 251).
        scale: Feature scale factor s (default 30.0 per paper).
        margin: Angular margin penalty m in radians (default 0.2 per paper).
        easy_margin: Whether to use easy margin fallback.
    """

    def __init__(
        self,
        in_features: int = 192,
        num_classes: int = 251,
        scale: float = 30.0,
        margin: float = 0.2,
        easy_margin: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError(f"Unsupported reduction '{reduction}'. Must be 'mean' or 'none'.")
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = float(scale)
        self.margin = float(margin)
        self.easy_margin = easy_margin
        self.reduction = reduction

        # Speaker prototype weight vectors (num_classes, in_features)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute trigonometric constants
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)  # -cos(m)
        self.mm = math.sin(math.pi - self.margin) * self.margin  # sin(m) * m

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute AAMSoftmax loss.
        
        Args:
            x: Input embeddings of shape (B, in_features).
            labels: Ground truth speaker label indices of shape (B,).
            
        Returns:
            Scalar cross-entropy loss with angular margin penalty on target class.
        """
        # Enforce fp32 execution per Rule 5.1 and ADR-007
        x_fp32 = x.float()
        w_fp32 = self.weight.float()

        # Cheap assert on label range when executing on CPU
        if labels.is_cpu:
            min_l, max_l = int(labels.min().item()), int(labels.max().item())
            assert min_l >= 0 and max_l < self.num_classes, (
                f"Labels out of range [0, {self.num_classes - 1}]: min={min_l}, max={max_l}"
            )

        # Normalize feature embeddings and class weights to unit hypersphere
        x_norm = F.normalize(x_fp32, p=2, dim=1)
        w_norm = F.normalize(w_fp32, p=2, dim=1)

        # Cosine similarity matrix cos(theta) = x_norm @ w_norm.T -> (B, num_classes)
        cos_theta = F.linear(x_norm, w_norm)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if self.margin == 0.0:
            output = cos_theta * self.scale
            return F.cross_entropy(output, labels, reduction=self.reduction)

        # cos(theta + m) = cos(theta) * cos(m) - sin(theta) * sin(m)
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta.pow(2), min=1e-7))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            # Monotonic fallback when theta + m > pi
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)

        # One-hot mask for ground truth labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Target class receives margin-penalized phi; impostor classes receive plain cos_theta
        # Sum in denominator is over all classes: exp(s*phi_{y_i}) + sum_{j != y_i} exp(s*cos(theta_j))
        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        output = output * self.scale

        loss = F.cross_entropy(output, labels, reduction=self.reduction)
        return loss

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return cosine similarity logits without margin for inference/evaluation."""
        x_norm = F.normalize(x.float(), p=2, dim=1)
        w_norm = F.normalize(self.weight.float(), p=2, dim=1)
        return F.linear(x_norm, w_norm)


__all__ = ["AAMSoftmax"]


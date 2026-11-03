"""Sliding-window local statistics computation for SKA-HNN.

Computes 1st-order (local mean) and 2nd-order (local std) statistics across
temporal frames using replicate padding to preserve sequence length T.

Follows Numerics Rule 5.1:
- Always computed in torch.float32.
- torch.clamp(var, min=1e-5) applied before torch.sqrt to prevent catastrophic cancellation.
- Safely handles constant/zero-variance inputs without NaN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalStatsPool1d(nn.Module):
    """Computes sliding-window local mean and std for temporal sequences (B, D, T).
    
    Attributes:
        window_size: Odd sliding window size (default W=21 frames).
        eps: Lower bound for variance clamping (default 1e-5).
        concat_input: If True, returns [X; mean; std] of shape (B, 3*D, T).
                      If False, returns [mean; std] of shape (B, 2*D, T).
    """

    def __init__(self, window_size: int = 21, eps: float = 1e-5, concat_input: bool = True) -> None:
        super().__init__()
        assert window_size % 2 == 1, f"window_size must be odd to preserve frame length, got {window_size}"
        self.window_size = window_size
        self.pad = window_size // 2
        self.eps = eps
        self.concat_input = concat_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local statistics over time dimension T.
        
        Args:
            x: Input tensor of shape (B, D, T).
            
        Returns:
            If concat_input=True: concatenated tensor (B, 3*D, T) in fp32.
            If concat_input=False: stats tensor (B, 2*D, T) in fp32.
        """
        # Rule 5.1: Local statistics must always be computed in torch.float32
        x_fp32 = x.float()

        # Replicate temporal boundary frames to avoid boundary edge collapse
        x_pad = F.pad(x_fp32, (self.pad, self.pad), mode="replicate")

        # Moving first moment (mean)
        mean = F.avg_pool1d(x_pad, kernel_size=self.window_size, stride=1)

        # Moving second moment (E[x^2])
        sq_mean = F.avg_pool1d(x_pad * x_pad, kernel_size=self.window_size, stride=1)

        # Variance with clamp >= eps to avoid negative values and catastrophic cancellation
        var = torch.clamp(sq_mean - mean * mean, min=self.eps)
        std = torch.sqrt(var)

        if self.concat_input:
            out = torch.cat([x_fp32, mean, std], dim=1)  # (B, 3*D, T)
        else:
            out = torch.cat([mean, std], dim=1)  # (B, 2*D, T)

        return out


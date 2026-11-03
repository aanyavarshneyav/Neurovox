"""2D Squeeze-and-Excitation (SE) block for 2D-CNN feature maps.

Pools across frequency and time dimensions to recalibrate channel responses.
"""

import torch
import torch.nn as nn


class SEBlock2D(nn.Module):
    """2D Squeeze-and-Excitation layer operating on the channel axis of (B, C, F, T)."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, C, F, T).
            
        Returns:
            Channel-reweighted tensor (B, C, F, T).
        """
        B, C, _, _ = x.shape
        # Global average pooling across Frequency (dim 2) and Time (dim 3) -> (B, C)
        s = x.mean(dim=(2, 3))
        g = self.fc(s).view(B, C, 1, 1)  # (B, C, 1, 1)
        return x * g


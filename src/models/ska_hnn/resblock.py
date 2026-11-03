"""Selective Kernel Attention ResBlock for SKA-HNN 2D-CNN frontend.

Architecture:
  x -> Conv2d(C, C, 3, p=1) -> BN2d -> ReLU
    -> [fwSKA -> cwSKA] (or Vanilla Conv2d for S5)
    -> SEBlock2D(C)
    -> Residual Skip (+) -> ReLU
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ska_hnn.cw_ska import ChannelWiseSKA
from src.models.ska_hnn.fw_ska import FrequencyWiseSKA
from src.models.ska_hnn.se import SEBlock2D


class SKAResBlock(nn.Module):
    """Residual block with Selective Kernel Attention and Squeeze-and-Excitation.
    
    Supports ablation switch S5 via use_ska=False (replaces fwSKA + cwSKA with a
    standard single-branch Conv2d + BN2d + ReLU layer).
    """

    def __init__(
        self,
        channels: int,
        freq_bins: int,
        reduction: int = 8,
        use_ska: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.freq_bins = freq_bins
        self.use_ska = use_ska

        # Initial 3x3 conv in residual branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        if use_ska:
            self.fw_ska = FrequencyWiseSKA(channels, freq_bins=freq_bins, reduction=reduction)
            self.cw_ska = ChannelWiseSKA(channels, reduction=reduction)
        else:
            # S5 Ablation: Vanilla 2D-CNN ResBlock without Selective Kernel Attention
            self.vanilla_conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )

        self.se = SEBlock2D(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, F, T).
            
        Returns:
            Residual output of shape (B, C, F, T).
        """
        residual = x
        out = self.conv1(x)

        if self.use_ska:
            out = self.fw_ska(out)
            out = self.cw_ska(out)
        else:
            out = self.vanilla_conv(out)

        out = self.se(out)
        return F.relu(out + residual, inplace=True)


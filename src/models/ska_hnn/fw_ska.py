"""Frequency-wise Selective Kernel Attention (fwSKA) module.

Recalibrates multi-scale 2D-CNN kernel branches along the frequency axis
by pooling across channel and temporal dimensions.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyWiseSKA(nn.Module):
    """Frequency-wise Selective Kernel Attention (fwSKA).
    
    Split: N=2 branches with 3x3 kernel (dilation=1) and 3x3 kernel (dilation=2).
    Fuse: Element-wise sum across branches.
    Select: Global average pooling over (Channel, Time) followed by a bottleneck MLP
            and softmax across branches for each frequency bin.
    """

    def __init__(self, channels: int, freq_bins: int, reduction: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.freq_bins = freq_bins
        d_f = max(freq_bins // reduction, 4)

        # Split: Branch 1 (3x3, dilation=1) and Branch 2 (3x3, dilation=2, effective 5x5 RF)
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Fuse & Squeeze: Bottleneck projection over frequency descriptor (B, F) -> (B, d_f)
        self.fc_squeeze = nn.Sequential(
            nn.Linear(freq_bins, d_f, bias=False),
            nn.BatchNorm1d(d_f),
            nn.ReLU(inplace=True),
        )

        # Select projection: (B, d_f) -> (B, 2 * freq_bins)
        self.fc_select = nn.Linear(d_f, 2 * freq_bins)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input feature map of shape (B, C, F, T).
            return_attn: If True, returns (output, attention_weights).
            
        Returns:
            Attended feature map of shape (B, C, F, T).
        """
        B, C, F_bins, T = x.shape
        assert F_bins == self.freq_bins, f"Expected freq_bins={self.freq_bins}, got {F_bins}"

        # 1. Split
        u1 = self.branch1(x)  # (B, C, F, T)
        u2 = self.branch2(x)  # (B, C, F, T)

        # 2. Fuse
        u_sum = u1 + u2  # (B, C, F, T)

        # Global average pooling across Channel (dim 1) and Time (dim 3) -> (B, F)
        s_fw = u_sum.mean(dim=(1, 3))

        # 3. Select
        z_fw = self.fc_squeeze(s_fw)  # (B, d_f)
        logits = self.fc_select(z_fw).view(B, 2, F_bins)  # (B, 2, F)

        # Softmax over branch dimension (dim=1)
        # dim=1: normalizes branch attention weights across the 2 kernel branches for each frequency bin
        attn = F.softmax(logits, dim=1)  # (B, 2, F)

        a1 = attn[:, 0, :].unsqueeze(1).unsqueeze(-1)  # (B, 1, F, 1)
        a2 = attn[:, 1, :].unsqueeze(1).unsqueeze(-1)  # (B, 1, F, 1)

        out = a1 * u1 + a2 * u2  # (B, C, F, T)

        if return_attn:
            return out, attn
        return out


"""Channel-wise Selective Kernel Attention (cwSKA) module.

Recalibrates multi-scale 2D-CNN kernel branches along the channel axis
by pooling across frequency and temporal dimensions.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseSKA(nn.Module):
    """Channel-wise Selective Kernel Attention (cwSKA).
    
    Split: N=2 branches with 3x3 kernel (dilation=1) and 3x3 kernel (dilation=2).
    Fuse: Element-wise sum across branches.
    Select: Global average pooling over (Frequency, Time) followed by a bottleneck MLP
            and softmax across branches for each channel.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.channels = channels
        d_c = max(channels // reduction, 4)

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

        # Fuse & Squeeze: Bottleneck projection over channel descriptor (B, C) -> (B, d_c)
        self.fc_squeeze = nn.Sequential(
            nn.Linear(channels, d_c, bias=False),
            nn.BatchNorm1d(d_c),
            nn.ReLU(inplace=True),
        )

        # Select projection: (B, d_c) -> (B, 2 * channels)
        self.fc_select = nn.Linear(d_c, 2 * channels)

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
        assert C == self.channels, f"Expected channels={self.channels}, got {C}"

        # 1. Split
        u1 = self.branch1(x)  # (B, C, F, T)
        u2 = self.branch2(x)  # (B, C, F, T)

        # 2. Fuse
        u_sum = u1 + u2  # (B, C, F, T)

        # Global average pooling across Frequency (dim 2) and Time (dim 3) -> (B, C)
        s_cw = u_sum.mean(dim=(2, 3))

        # 3. Select
        z_cw = self.fc_squeeze(s_cw)  # (B, d_c)
        logits = self.fc_select(z_cw).view(B, 2, C)  # (B, 2, C)

        # Softmax over branch dimension (dim=1)
        # dim=1: normalizes branch attention weights across the 2 kernel branches for each channel
        attn = F.softmax(logits, dim=1)  # (B, 2, C)

        a1 = attn[:, 0, :].unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        a2 = attn[:, 1, :].unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        out = a1 * u1 + a2 * u2  # (B, C, F, T)

        if return_attn:
            return out, attn
        return out


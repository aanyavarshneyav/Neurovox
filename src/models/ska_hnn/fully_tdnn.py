"""Fully-TDNN module for SKA-HNN.

Cascades 3 dilated 1D convolutional layers (dilations 1, 2, 3) with batch normalization
and ReLU activations to model multi-scale temporal context.
"""

import torch
import torch.nn as nn


class Conv1dReLUBN(nn.Module):
    """Standard 1D TDNN block: Conv1d + ReLU + BatchNorm1d."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FullyTDNNModule(nn.Module):
    """Fully-TDNN Module: 3 cascaded Conv1d+ReLU+BN layers with dilations 1, 2, and 3.
    
    Accepts concatenated frame features with local statistics [X_l; LS(X_l)] of shape
    (B, in_dim, T) and outputs representation X_t of shape (B, out_dim, T).
    """

    def __init__(self, in_dim: int, out_dim: int = 256) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            Conv1dReLUBN(in_dim, out_dim, kernel_size=3, dilation=1),
            Conv1dReLUBN(out_dim, out_dim, kernel_size=3, dilation=2),
            Conv1dReLUBN(out_dim, out_dim, kernel_size=3, dilation=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, in_dim, T).
            
        Returns:
            X_t feature tensor of shape (B, out_dim, T).
        """
        return self.net(x)


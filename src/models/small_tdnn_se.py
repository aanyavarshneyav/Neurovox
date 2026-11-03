"""Lightweight 1D-CNN with Squeeze-and-Excitation for Phase 2A pipeline validation.

Note: This is a fast, cheap baseline architecture ('small_tdnn_se') used strictly
for pipeline, loss, trainer, and extractor validation. It is NOT ECAPA-TDNN (the
parameter-matched ECAPA baseline is implemented in Phase 2C).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1d(nn.Module):
    """Channel-wise Squeeze-and-Excitation block for 1D temporal representations."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = x.mean(dim=2)  # (B, C)
        w = self.fc(s).unsqueeze(2)  # (B, C, 1)
        return x * w


class TDNNBlock(nn.Module):
    """1D Dilated Conv with BatchNorm, ReLU, Squeeze-and-Excitation, and Residual Skip."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.se = SEBlock1d(out_channels)
        self.has_residual = (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.has_residual else None
        out = self.relu(self.bn(self.conv(x)))
        out = self.se(out)
        if res is not None:
            out = out + res
        return out


class StatsPool(nn.Module):
    """Statistical temporal pooling (mean and standard deviation) with numerical guardrails."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # Always compute statistics in fp32 per Rule 5.1
        orig_dtype = x.dtype
        x_fp32 = x.float()

        mean = x_fp32.mean(dim=2)  # (B, C)
        var = torch.clamp((x_fp32 - mean.unsqueeze(2)).pow(2).mean(dim=2), min=self.eps)
        std = torch.sqrt(var)  # (B, C)

        out = torch.cat([mean, std], dim=1).to(orig_dtype)  # (B, 2*C)
        return out


class SmallTDNNSE(nn.Module):
    """Small TDNN with Squeeze-and-Excitation for Phase 2A sanity checks.
    
    Accepts log-Mel filterbank spectrogram (B, 80, T) and outputs 192-dim speaker embedding.
    """

    def __init__(self, in_features: int = 80, channels: int = 128, emb_dim: int = 192) -> None:
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_features, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        self.block1 = TDNNBlock(channels, channels, kernel_size=3, dilation=2)
        self.block2 = TDNNBlock(channels, channels, kernel_size=3, dilation=3)
        self.block3 = TDNNBlock(channels, channels, kernel_size=3, dilation=4)

        # Multi-layer aggregation
        self.mfa = nn.Sequential(
            nn.Conv1d(channels * 3, channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),
        )

        self.pool = StatsPool(eps=1e-5)
        self.bn_pool = nn.BatchNorm1d(channels * 4)

        self.fc = nn.Linear(channels * 4, emb_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input acoustic tensor (B, 80, T) or (B, 1, 80, T).
            
        Returns:
            Speaker embedding (B, emb_dim).
        """
        if x.ndim == 4:
            # Squeeze channel dimension if present: (B, 1, F, T) -> (B, F, T)
            x = x.squeeze(1)

        h0 = self.in_conv(x)
        h1 = self.block1(h0)
        h2 = self.block2(h1)
        h3 = self.block3(h2)

        # Multi-scale temporal aggregation
        cat = torch.cat([h1, h2, h3], dim=1)
        mfa_out = self.mfa(cat)

        # Stats pooling over time
        pooled = self.pool(mfa_out)
        pooled_bn = self.bn_pool(pooled)

        # Embedding projection
        emb = self.bn_out(self.fc(pooled_bn))
        return emb


def small_tdnn_se(in_features: int = 80, channels: int = 128, emb_dim: int = 192) -> SmallTDNNSE:
    """Factory function for SmallTDNNSE model."""
    return SmallTDNNSE(in_features=in_features, channels=channels, emb_dim=emb_dim)


__all__ = ["SmallTDNNSE", "small_tdnn_se"]


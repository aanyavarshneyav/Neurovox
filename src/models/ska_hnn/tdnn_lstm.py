"""TDNN-LSTM hybrid module for SKA-HNN.

Cascades a 1D TDNN (dilated Conv1d + ReLU + BN1d) with a unidirectional LSTM
to capture both short-term acoustic context and long-term speaker characteristics.
"""

import torch
import torch.nn as nn


class TDNNLSTMModule(nn.Module):
    """TDNN-LSTM Hybrid Module.
    
    Accepts concatenated frame features with local statistics [X_c; LS(X_c)] of shape
    (B, in_dim, T) and outputs representation X_l of shape (B, out_dim, T).
    """

    def __init__(self, in_dim: int, out_dim: int = 256) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 1D TDNN layer with kernel_size=3, padding=1 (dilation=1)
        self.tdnn = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_dim),
        )

        # Unidirectional single-layer LSTM
        self.lstm = nn.LSTM(
            input_size=out_dim,
            hidden_size=out_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, in_dim, T).
            
        Returns:
            X_l feature tensor of shape (B, out_dim, T).
        """
        # TDNN feature extraction: (B, in_dim, T) -> (B, out_dim, T)
        h = self.tdnn(x)

        # Transpose to (B, T, out_dim) for batch_first LSTM
        h_seq = h.transpose(1, 2).contiguous()
        lstm_out, _ = self.lstm(h_seq)  # (B, T, out_dim)

        # Transpose back to (B, out_dim, T)
        lstm_out = lstm_out.transpose(1, 2).contiguous()
        return self.bn(self.relu(lstm_out))


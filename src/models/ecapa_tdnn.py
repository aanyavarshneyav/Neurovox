"""ECAPA-TDNN baseline architecture for speaker verification.

Reference:
    B. Desplanques, J. Thienpondt, and K. Demuynck, "ECAPA-TDNN: Emphasized Channel
    Attention, Propagation and Aggregation in TDNN Based Speaker Verification",
    Interspeech 2020.

Components:
    1. 1D TDNN Initial Conv: Conv1d(in_features=80, C, k=5, p=2) + BN + ReLU
    2. 3x Squeeze-and-Excitation Res2Net (SERes2Net) Blocks with dilations [2, 3, 4]
       and Res2Net scale 8.
    3. Multi-layer Feature Aggregation (MFA): Conv1d(3*C, 3*C, k=1) + BN + ReLU
    4. Attentive Statistics Pooling (ASP) with channel attention and global context
       computing weighted mean and std -> (B, 2*3*C).
    5. Linear projection to 192-dimensional speaker embedding.

Parameter Configurations:
    - Parameter-matched (scaled SKA-HNN baseline, C=400, MFA=1200):
        Total params: 4,123,210 | Trainable: 4,123,210
        Difference from SKA-HNN (4,167,700): -44,490 (-1.07%)
    - Standard ECAPA-TDNN (SpeechBrain standard, C=1024, MFA=3072):
        Total params: 20,767,552 | Trainable: 20,767,552
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN as SpeechBrainECAPA


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN model conforming to Neurovox input/output tensor contracts.

    Accepts log-Mel filterbank spectrogram (B, 80, T) or (B, 1, 80, T)
    and extracts a 192-dimensional speaker embedding of shape (B, emb_dim).
    """

    def __init__(
        self,
        in_features: int = 80,
        channels: Union[int, List[int]] = 400,
        mfa_dim: Optional[int] = None,
        emb_dim: int = 192,
        res2net_scale: int = 8,
        se_channels: int = 128,
        attention_channels: int = 128,
        global_context: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.emb_dim = emb_dim

        if isinstance(channels, int):
            mfa = mfa_dim if mfa_dim is not None else channels * 3
            self.channel_config = [channels, channels, channels, channels, mfa]
        elif isinstance(channels, (list, tuple)):
            if len(channels) != 5:
                raise ValueError(
                    f"channels list must contain 5 integers [conv, b1, b2, b3, mfa], got {channels}"
                )
            self.channel_config = list(channels)
        else:
            raise TypeError(f"Unsupported type for channels: {type(channels)}")

        self.ecapa = SpeechBrainECAPA(
            input_size=in_features,
            lin_neurons=emb_dim,
            channels=self.channel_config,
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=attention_channels,
            res2net_scale=res2net_scale,
            se_channels=se_channels,
            global_context=global_context,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting speaker embedding.

        Args:
            x: Acoustic input tensor of shape (B, 80, T), (B, 1, 80, T), or (B, T, 80).

        Returns:
            Speaker embedding of shape (B, emb_dim).
        """
        # Squeeze 4D channel dimension if present: (B, 1, F, T) -> (B, F, T)
        if x.ndim == 4:
            x = x.squeeze(1)

        # Transpose from (B, F, T) to (B, T, F) if feature dimension is in dim 1
        if x.shape[1] == self.in_features and x.shape[2] != self.in_features:
            x = x.transpose(1, 2)

        # Forward through SpeechBrain ECAPA: returns (B, 1, emb_dim)
        out = self.ecapa(x)

        # Return embedding of shape (B, emb_dim)
        return out.squeeze(1)


def ecapa_tdnn_param_matched(
    in_features: int = 80,
    emb_dim: int = 192,
) -> ECAPATDNN:
    """Factory for parameter-matched ECAPA-TDNN (C=400, 4,123,210 params vs SKA-HNN 4,167,700)."""
    return ECAPATDNN(
        in_features=in_features,
        channels=[400, 400, 400, 400, 1200],
        emb_dim=emb_dim,
    )


def ecapa_tdnn_standard(
    in_features: int = 80,
    emb_dim: int = 192,
) -> ECAPATDNN:
    """Factory for standard SpeechBrain ECAPA-TDNN (C=1024, 20,767,552 params)."""
    return ECAPATDNN(
        in_features=in_features,
        channels=[1024, 1024, 1024, 1024, 3072],
        emb_dim=emb_dim,
    )


__all__ = [
    "ECAPATDNN",
    "ecapa_tdnn_param_matched",
    "ecapa_tdnn_standard",
]

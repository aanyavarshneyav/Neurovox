"""Multi-Level Attentive Statistics Pooling (MLASP) for SKA-HNN.

Aggregates frame-level representations into a fixed-dimensional utterance representation
using channel-dependent attentive first- and second-order statistics.

Follows Numerics Rule 5.1:
- Local context and weighted statistics computed in fp32 explicitly.
- Clamps variance >= 1e-5 before torch.sqrt.
- Softmax over temporal frames with explicit dim=2.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ska_hnn.local_stats import LocalStatsPool1d


class MLASP(nn.Module):
    """Multi-Level Attentive Statistics Pooling (MLASP).
    
    Supports ablation switches:
      - S0, S1, S3, S5, S6 (use_ls_context=True): Attention computed over X_i^c = [X_i; mu_i; sigma_i].
      - S2, S4 (use_ls_context=False): Attention computed directly over X_i.
    """

    def __init__(
        self,
        feat_dim: int,
        attn_dim: int = 128,
        use_ls_context: bool = True,
        window_size: int = 21,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.attn_dim = attn_dim
        self.use_ls_context = use_ls_context
        self.eps = eps

        if use_ls_context:
            self.ls_pool = LocalStatsPool1d(window_size=window_size, eps=eps, concat_input=True)
            in_attn_dim = 3 * feat_dim
        else:
            self.ls_pool = None
            in_attn_dim = feat_dim

        # Attention bottleneck over X_i^c (or X_i) -> reduced to match X_i dimension (feat_dim)
        self.attention = nn.Sequential(
            nn.Conv1d(in_attn_dim, attn_dim, kernel_size=1, bias=True),
            nn.Tanh(),
            nn.BatchNorm1d(attn_dim),
            nn.Conv1d(attn_dim, feat_dim, kernel_size=1, bias=True),
        )

        # Batch normalization over pooled utterance representation (mean & std concatenated)
        self.out_bn = nn.BatchNorm1d(2 * feat_dim)

    def forward(
        self, x_i: torch.Tensor, return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x_i: Combined frame representations of shape (B, feat_dim, T).
            return_attn: If True, returns (pooled, attention_weights).
            
        Returns:
            Pooled utterance representation of shape (B, 2 * feat_dim).
        """
        # 1. Obtain context tensor for attention scoring
        if self.use_ls_context:
            assert self.ls_pool is not None
            x_ic = self.ls_pool(x_i)  # (B, 3*feat_dim, T)
        else:
            x_ic = x_i  # (B, feat_dim, T)

        # 2. Compute channel-dependent attention logits: (B, feat_dim, T)
        attn_logits = self.attention(x_ic)

        # Softmax over time dimension (dim=2)
        # dim=2: normalizes attention weights over the time frames T for each channel dimension
        alpha = F.softmax(attn_logits, dim=2)  # (B, feat_dim, T)

        # 3. Attentive pooling in fp32 for numerical safety (Rule 5.1)
        orig_dtype = x_i.dtype
        x_fp32 = x_i.float()
        alpha_fp32 = alpha.float()

        # Weighted first moment (mean): sum over time frames T (dim=2)
        mean = torch.sum(alpha_fp32 * x_fp32, dim=2)  # (B, feat_dim)

        # Weighted second moment (E[x^2]): sum over time frames T (dim=2)
        sq_mean = torch.sum(alpha_fp32 * (x_fp32 * x_fp32), dim=2)  # (B, feat_dim)

        # Variance with clamp >= eps to prevent catastrophic cancellation and NaNs
        var = torch.clamp(sq_mean - mean * mean, min=self.eps)
        std = torch.sqrt(var)  # (B, feat_dim)

        # Utterance-level concatenation: (B, 2*feat_dim)
        pooled = torch.cat([mean, std], dim=1).to(orig_dtype)
        out = self.out_bn(pooled)

        if return_attn:
            return out, alpha
        return out


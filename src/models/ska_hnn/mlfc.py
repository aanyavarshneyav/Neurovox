"""Multi-Level Feature Combination (MLFC) module for SKA-HNN.

Aggregates frame representations from the 2D-CNN (X_c), TDNN-LSTM (X_l),
and Fully-TDNN (X_t) stages to form composite representations X_i.
"""

from typing import Optional
import torch
import torch.nn as nn


class MultiLevelFeatureCombination(nn.Module):
    """Combines representations across network depths.
    
    Supports ablation switches:
      - S0, S1, S2, S5: X_i = [X_c; X_l; X_t]
      - S3, S4 (use_mlfc=False): X_i = X_t
      - S6 (no 2D-CNN frontend): X_i = [X_l; X_t]
    """

    def __init__(self, use_mlfc: bool = True) -> None:
        super().__init__()
        self.use_mlfc = use_mlfc

    def forward(
        self,
        x_t: torch.Tensor,
        x_l: Optional[torch.Tensor] = None,
        x_c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combine feature representations.
        
        Args:
            x_t: Output of Fully-TDNN module (B, C1, T).
            x_l: Output of TDNN-LSTM module (B, C1, T) or None.
            x_c: Output of 2D-CNN frontend (B, d_c, T) or None.
            
        Returns:
            X_i of shape (B, d_i, T).
        """
        if not self.use_mlfc:
            # S3 / S4: MLFC disabled, feed Fully-TDNN output directly
            return x_t

        if x_c is None:
            # S6: 2D-CNN disabled, combine TDNN-LSTM and Fully-TDNN representations
            assert x_l is not None, "x_l must be provided when use_mlfc is True"
            return torch.cat([x_l, x_t], dim=1)

        # Full proposed combination: [X_c; X_l; X_t]
        assert x_l is not None, "x_l must be provided when use_mlfc is True"
        return torch.cat([x_c, x_l, x_t], dim=1)


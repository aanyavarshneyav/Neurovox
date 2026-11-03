"""Selective Kernel Attention-driven Hybrid Neural Network (SKA-HNN).

Reference Paper:
  J. Alam, A. Fathan, M. S. Alam, "A Hybrid Neural Approach to Speaker Verification
  with an Improved Additive Angular Margin Loss", IJCNN 2025.

Assembles:
  1. 2D-CNN Frontend (Preceding Conv2d -> 2x SKAResBlock -> Succeeding Conv2d)
  2. Local Statistics 1 -> TDNN-LSTM Hybrid Module
  3. Local Statistics 2 -> Fully-TDNN Module
  4. Multi-Level Feature Combination (MLFC)
  5. Multi-Level Attentive Statistics Pooling (MLASP) with Local Statistics Context
  6. Two FC+BN layers (FC1 primary embedding closest to pooling; FC2 diagnostic)

Supports Table II ablation switches S0, S1, S2, S3, S4, S5, S6.
Scaled setting: C=32, C1=256, attn_dim=128, emb_dim=192.
Full paper setting: C=128, C1=512, attn_dim=256, emb_dim=192.
"""

from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ska_hnn.fully_tdnn import FullyTDNNModule
from src.models.ska_hnn.local_stats import LocalStatsPool1d
from src.models.ska_hnn.mlasp import MLASP
from src.models.ska_hnn.mlfc import MultiLevelFeatureCombination
from src.models.ska_hnn.resblock import SKAResBlock
from src.models.ska_hnn.tdnn_lstm import TDNNLSTMModule


class SKAHNN(nn.Module):
    """Selective Kernel Attention-driven Hybrid Neural Network (SKA-HNN).
    
    Accepts acoustic Mel-filterbank spectrograms of shape (B, 80, T) or (B, 1, 80, T)
    and extracts a 192-dimensional speaker embedding.
    """

    def __init__(
        self,
        n_mels: int = 80,
        c_cnn: int = 32,
        c_tdnn: int = 256,
        emb_dim: int = 192,
        attn_dim: int = 128,
        ls_window: int = 21,
        frontend: str = "ska_2dcnn",  # "ska_2dcnn", "vanilla_2dcnn", "none"
        use_mlfc: bool = True,         # False for S3, S4
        use_ls_context: bool = True,   # False for S2, S4
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.c_cnn = c_cnn
        self.c_tdnn = c_tdnn
        self.emb_dim = emb_dim
        self.attn_dim = attn_dim
        self.ls_window = ls_window
        self.frontend = frontend.lower()
        self.use_mlfc = use_mlfc
        self.use_ls_context = use_ls_context

        # Validate frontend configuration
        assert self.frontend in {"ska_2dcnn", "vanilla_2dcnn", "none"}, (
            f"Invalid frontend mode '{frontend}'. Expected 'ska_2dcnn', 'vanilla_2dcnn', or 'none'."
        )

        # -------------------------------------------------------------
        # 1. 2D-CNN Frontend
        # -------------------------------------------------------------
        if self.frontend in {"ska_2dcnn", "vanilla_2dcnn"}:
            use_ska = (self.frontend == "ska_2dcnn")
            # Preceding Conv2d: (B, 1, F=80, T) -> (B, C, F=40, T)
            self.conv2d_in = nn.Sequential(
                nn.Conv2d(1, c_cnn, kernel_size=3, stride=(2, 1), padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(c_cnn),
            )
            # 2x ResBlocks operating at frequency resolution 40
            self.resblock1 = SKAResBlock(c_cnn, freq_bins=n_mels // 2, reduction=8, use_ska=use_ska)
            self.resblock2 = SKAResBlock(c_cnn, freq_bins=n_mels // 2, reduction=8, use_ska=use_ska)

            # Succeeding Conv2d: (B, C, F=40, T) -> (B, C, F=20, T)
            self.conv2d_out = nn.Sequential(
                nn.Conv2d(c_cnn, c_cnn, kernel_size=3, stride=(2, 1), padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(c_cnn),
            )
            # Flatten frequency and channel: d_c = c_cnn * (80 // 4) = 20 * c_cnn
            self.d_c = c_cnn * (n_mels // 4)
        else:
            # S6: No 2D-CNN frontend. Acoustic features feed directly to TDNN-LSTM.
            self.conv2d_in = None
            self.resblock1 = None
            self.resblock2 = None
            self.conv2d_out = None
            self.d_c = n_mels

        # -------------------------------------------------------------
        # 2. Frame-level Cascade: TDNN-LSTM Module
        # -------------------------------------------------------------
        # Local statistics 1 over X_c
        self.ls_c = LocalStatsPool1d(window_size=ls_window, concat_input=True)
        # Input to TDNN-LSTM is [X_c; LS(X_c)] of dimension 3 * d_c
        self.tdnn_lstm = TDNNLSTMModule(in_dim=3 * self.d_c, out_dim=c_tdnn)

        # -------------------------------------------------------------
        # 3. Frame-level Cascade: Fully-TDNN Module
        # -------------------------------------------------------------
        # Local statistics 2 over X_l
        self.ls_l = LocalStatsPool1d(window_size=ls_window, concat_input=True)
        # Input to Fully-TDNN is [X_l; LS(X_l)] of dimension 3 * c_tdnn
        self.fully_tdnn = FullyTDNNModule(in_dim=3 * c_tdnn, out_dim=c_tdnn)

        # -------------------------------------------------------------
        # 4. Multi-Level Feature Combination (MLFC)
        # -------------------------------------------------------------
        self.mlfc = MultiLevelFeatureCombination(use_mlfc=use_mlfc)
        if not use_mlfc:
            # S3 & S4: X_i = X_t only
            self.d_i = c_tdnn
        elif self.frontend == "none":
            # S6: X_i = [X_l; X_t]
            self.d_i = 2 * c_tdnn
        else:
            # S0, S1, S2, S5: X_i = [X_c; X_l; X_t]
            self.d_i = self.d_c + 2 * c_tdnn

        # -------------------------------------------------------------
        # 5. Multi-Level Attentive Statistics Pooling (MLASP)
        # -------------------------------------------------------------
        self.mlasp = MLASP(
            feat_dim=self.d_i,
            attn_dim=attn_dim,
            use_ls_context=use_ls_context,
            window_size=ls_window,
        )

        # -------------------------------------------------------------
        # 6. Classifier / Head: Two FC+BN Layers
        # -------------------------------------------------------------
        # FC1: Primary embedding layer closest to pooling (Paper Sec. II-D, ADR-004)
        self.fc1 = nn.Sequential(
            nn.Linear(2 * self.d_i, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )
        # FC2: Second FC layer before softmax (Paper Table VIII, ADR-004 diagnostic)
        self.fc2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, x: torch.Tensor, extract_layer: str = "auto") -> torch.Tensor:
        """Extract speaker embedding from input acoustic spectrogram.
        
        Args:
            x: Acoustic input tensor of shape (B, 80, T) or (B, 1, 80, T).
            extract_layer: "auto" (default: FC2 in training mode, FC1 in eval mode),
                           "fc1" (explicit primary embedding closest to pooling),
                           "fc2" (explicit diagnostic embedding).
                           
        Returns:
            Speaker embedding of shape (B, emb_dim).
        """
        if x.ndim == 3:
            # (B, F, T) -> (B, 1, F, T) for 2D convolutions
            mfb = x
            x_2d = x.unsqueeze(1)
        elif x.ndim == 4:
            # (B, 1, F, T)
            mfb = x.squeeze(1)
            x_2d = x
        else:
            raise ValueError(f"Expected 3D (B, F, T) or 4D (B, 1, F, T) input, got ndim={x.ndim}")

        B, _, T = mfb.shape

        # 1. 2D-CNN Frontend
        if self.frontend != "none":
            h = self.conv2d_in(x_2d)       # (B, C, 40, T)
            h = self.resblock1(h)          # (B, C, 40, T)
            h = self.resblock2(h)          # (B, C, 40, T)
            h = self.conv2d_out(h)         # (B, C, 20, T)
            x_c = h.reshape(B, self.d_c, T)  # (B, d_c, T)
        else:
            x_c = None

        # 2. TDNN-LSTM Hybrid Module
        x_c_in = mfb if x_c is None else x_c
        x_c_aug = self.ls_c(x_c_in)         # (B, 3*d_c, T)
        x_l = self.tdnn_lstm(x_c_aug)       # (B, c_tdnn, T)

        # 3. Fully-TDNN Module
        x_l_aug = self.ls_l(x_l)            # (B, 3*c_tdnn, T)
        x_t = self.fully_tdnn(x_l_aug)      # (B, c_tdnn, T)

        # 4. Multi-Level Feature Combination (MLFC)
        x_i = self.mlfc(x_t=x_t, x_l=x_l, x_c=x_c)  # (B, d_i, T)

        # 5. Multi-Level Attentive Statistics Pooling (MLASP)
        utt_vec = self.mlasp(x_i)           # (B, 2*d_i)

        # 6. Classification Network Embeddings
        emb1 = self.fc1(utt_vec)            # (B, emb_dim)

        if extract_layer == "fc1":
            return emb1

        if extract_layer == "fc2":
            return self.fc2(emb1)

        # "auto" mode:
        # In eval mode (extractor.py / validation), return FC1 (Paper Sec II-D / Fig 3 caption:
        # "embeddings are extracted from the FC layer closest to the pooling layer").
        # In training mode, return FC2 so the entire network (including FC2) trains with gradients.
        if self.training:
            return self.fc2(emb1)
        return emb1


def build_ska_hnn(
    switch: str = "S0",
    c_cnn: int = 32,
    c_tdnn: int = 256,
    emb_dim: int = 192,
    attn_dim: int = 128,
    ls_window: int = 21,
    n_mels: int = 80,
    **kwargs: Any,
) -> SKAHNN:
    """Build SKA-HNN model configured for ablation switches S0–S6.
    
    Mapping follows Table II and Section IV-B of Alam et al. (IJCNN 2025):
      - S0: Full Proposed (SKA-2DCNN + MLFC + MLASP Context + IAAMSoftmax)
      - S1: Standard Margin (SKA-2DCNN + MLFC + MLASP Context + AAMSoftmax)
      - S2: No MLASP Context (SKA-2DCNN + MLFC + No Context + AAMSoftmax)
      - S3: No MLFC (SKA-2DCNN + No MLFC + MLASP Context + AAMSoftmax)
      - S4: No MLFC & No Context (SKA-2DCNN + No MLFC + No Context + AAMSoftmax)
      - S5: Vanilla 2D-CNN (Vanilla-2DCNN + MLFC + MLASP Context + AAMSoftmax)
      - S6: No 2D-CNN Frontend (No Frontend + MLFC + MLASP Context + AAMSoftmax)
    """
    switch = switch.upper()
    valid_switches = {"S0", "S1", "S2", "S3", "S4", "S5", "S6"}
    if switch not in valid_switches:
        raise ValueError(f"Unknown ablation switch '{switch}'. Valid options: {sorted(valid_switches)}")

    # Default settings for S0 / S1
    frontend = "ska_2dcnn"
    use_mlfc = True
    use_ls_context = True

    if switch == "S2":
        use_ls_context = False
    elif switch == "S3":
        use_mlfc = False
        use_ls_context = True
    elif switch == "S4":
        use_mlfc = False
        use_ls_context = False
    elif switch == "S5":
        frontend = "vanilla_2dcnn"
    elif switch == "S6":
        frontend = "none"

    # Allow explicit overrides from kwargs if passed
    frontend = kwargs.get("frontend", frontend)
    use_mlfc = kwargs.get("use_mlfc", use_mlfc)
    use_ls_context = kwargs.get("use_ls_context", use_ls_context)

    return SKAHNN(
        n_mels=n_mels,
        c_cnn=c_cnn,
        c_tdnn=c_tdnn,
        emb_dim=emb_dim,
        attn_dim=attn_dim,
        ls_window=ls_window,
        frontend=frontend,
        use_mlfc=use_mlfc,
        use_ls_context=use_ls_context,
    )


def build_ska_hnn_from_config(cfg: Any) -> SKAHNN:
    """Build SKA-HNN directly from a ConfigDict or nested dict."""
    model_cfg = cfg.get("model", {})
    audio_cfg = cfg.get("audio", {})

    switch = model_cfg.get("switch", "S0")
    c_cnn = int(model_cfg.get("c_cnn", model_cfg.get("channels", 32)))
    c_tdnn = int(model_cfg.get("c_tdnn", 256))
    emb_dim = int(model_cfg.get("emb_dim", 192))
    attn_dim = int(model_cfg.get("attn_dim", 128))
    ls_window = int(model_cfg.get("ls_window", 21))
    n_mels = int(audio_cfg.get("n_mels", 80))

    # Optional explicit switch fields from config
    overrides = {}
    if "frontend" in model_cfg:
        overrides["frontend"] = str(model_cfg["frontend"])
    if "use_mlfc" in model_cfg:
        overrides["use_mlfc"] = bool(model_cfg["use_mlfc"])
    if "use_ls_context" in model_cfg:
        overrides["use_ls_context"] = bool(model_cfg["use_ls_context"])

    return build_ska_hnn(
        switch=switch,
        c_cnn=c_cnn,
        c_tdnn=c_tdnn,
        emb_dim=emb_dim,
        attn_dim=attn_dim,
        ls_window=ls_window,
        n_mels=n_mels,
        **overrides,
    )


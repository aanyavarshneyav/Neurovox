"""Improved Additive Angular Margin Softmax (IAAMSoftmax) with AdaptiveDrop.

Reference Paper:
    J. Alam, A. Fathan, M. S. Alam, "A Hybrid Neural Approach to Speaker Verification
    with an Improved Additive Angular Margin Loss", IJCNN 2025, Section III & Algorithm 1.

Provenance Tags:
    - [Paper]: s=30.0, m=0.2, tau=0.423 (cos 65 deg), warm-up ESD=5, Algorithm 1 sequence.
    - [Engineering Choice]: ESC=5 (ADR-001); j != y_i target summation fix; keep if CS >= tau;
      deterministic minimum keep-ratio safeguard (min_keep_ratio=0.5 with top-k fallback).
    - [External-UNVERIFIED] / [Engineering Choice]: tau_corr=0.5; label correction condition:
      if CS(x, y) < tau and max_{c != y} CS(x, c) >= tau_corr, rectify label to argmax_{c != y} CS(x, c).

Algorithm 1 Sequence (Alam et al., IJCNN 2025):
    1. Compute cosine similarities CS(x_i, c) on the fly between L2-normalized embeddings
       and L2-normalized class prototypes in fp32.
    2. Stage 1 — Label Correction (epoch >= ESC):
       Suspicious samples with CS(x_i, y_i) < tau are inspected. If the best competing class
       satisfies max_{c != y_i} CS(x_i, c) >= tau_corr, the label is rectified to that class.
       This alleviates over-cleaning by correcting mislabeled samples before filtering.
    3. Stage 2 — Sample Filtering (epoch >= ESD):
       Evaluate similarity to (possibly rectified) label: CS(x_i, y_i').
       Samples with CS(x_i, y_i') >= tau are retained; samples with CS < tau are dropped.
    4. Safety Guard (ADR-001):
       If retained samples < max(1, ceil(B * min_keep_ratio)), top-k samples by CS are kept
       to prevent empty-batch gradient collapse. Guard activation is logged.
    5. AAMSoftmax Loss:
       Cross-entropy with angular margin penalty m=0.2 and scale s=30.0 computed strictly
       over retained samples. Dropped samples receive zero gradient.
"""

import math
import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.aam_softmax import AAMSoftmax

logger = logging.getLogger(__name__)


class IAAMSoftmax(AAMSoftmax):
    """Improved Additive Angular Margin Softmax with AdaptiveDrop.

    Args:
        in_features: Dimension of input speaker embeddings (default: 192).
        num_classes: Number of training speaker classes (default: 251).
        scale: Feature scale factor s (default: 30.0 per paper).
        margin: Angular margin penalty m in radians (default: 0.2 per paper).
        easy_margin: Whether to use easy margin fallback (default: False).
        reduction: Loss reduction mode ('mean' or 'none', default: 'mean').
        tau: Cosine similarity filtering threshold (default: 0.423 per paper).
        tau_corr: Cosine similarity label correction threshold (default: 0.5, ADR-001).
        esd: Epoch to start dropping/filtering samples (default: 5 per paper).
        esc: Epoch to start correcting labels (default: 5, ADR-001).
        min_keep_ratio: Minimum fraction of batch retained by safeguard (default: 0.5, ADR-001).
        use_adaptive_drop: Master toggle for AdaptiveDrop mechanics (default: True).
    """

    def __init__(
        self,
        in_features: int = 192,
        num_classes: int = 251,
        scale: float = 30.0,
        margin: float = 0.2,
        easy_margin: bool = False,
        reduction: str = "mean",
        tau: float = 0.423,
        tau_corr: float = 0.5,
        esd: int = 5,
        esc: int = 5,
        min_keep_ratio: float = 0.5,
        use_adaptive_drop: bool = True,
    ) -> None:
        super().__init__(
            in_features=in_features,
            num_classes=num_classes,
            scale=scale,
            margin=margin,
            easy_margin=easy_margin,
            reduction=reduction,
        )
        if not (0.0 <= float(min_keep_ratio) <= 1.0):
            raise ValueError(f"min_keep_ratio must be in [0.0, 1.0], got {min_keep_ratio}")
        if not (-1.0 <= float(tau) <= 1.0):
            raise ValueError(f"tau must be in [-1.0, 1.0], got {tau}")
        if not (-1.0 <= float(tau_corr) <= 1.0):
            raise ValueError(f"tau_corr must be in [-1.0, 1.0], got {tau_corr}")
        if int(esd) < 1:
            raise ValueError(f"esd must be >= 1, got {esd}")
        if int(esc) < 1:
            raise ValueError(f"esc must be >= 1, got {esc}")

        self.tau = float(tau)
        self.tau_corr = float(tau_corr)
        self.esd = int(esd)
        self.esc = int(esc)
        self.min_keep_ratio = float(min_keep_ratio)
        self.use_adaptive_drop = bool(use_adaptive_drop)

        # 1-indexed epoch counter (default 1)
        self.current_epoch: int = 1

        # Diagnostics for most recent batch
        self.last_diagnostics: Dict[str, Any] = {}
        self.last_keep_mask: Optional[torch.Tensor] = None

        # Epoch-level diagnostic accumulators
        self.reset_epoch_diagnostics()

    def set_epoch(self, epoch: int) -> None:
        """Set the 1-indexed training epoch counter and reset epoch accumulators."""
        if epoch < 1:
            raise ValueError(f"Epoch must be >= 1, got {epoch}")
        if epoch != self.current_epoch:
            self.reset_epoch_diagnostics()
        self.current_epoch = epoch

    def reset_epoch_diagnostics(self) -> None:
        """Reset epoch-level metric accumulators."""
        self._epoch_total_samples: int = 0
        self._epoch_retained_samples: int = 0
        self._epoch_filtered_samples: int = 0
        self._epoch_corrected_samples: int = 0
        self._epoch_guard_activations: int = 0
        self._epoch_batches: int = 0

    def get_last_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic metrics dictionary from the most recent forward pass."""
        return dict(self.last_diagnostics)

    def get_epoch_diagnostics(self) -> Dict[str, Any]:
        """Return aggregated epoch diagnostics across all batches since last reset."""
        total = max(1, self._epoch_total_samples)
        keep_ratio = float(self._epoch_retained_samples) / total
        return {
            "epoch": self.current_epoch,
            "loss_num_corrected": self._epoch_corrected_samples,
            "loss_num_filtered": self._epoch_filtered_samples,
            "loss_num_retained": self._epoch_retained_samples,
            "loss_total_samples": self._epoch_total_samples,
            "loss_keep_ratio": round(keep_ratio, 4),
            "loss_guard_activations": self._epoch_guard_activations,
            "loss_batches": self._epoch_batches,
        }

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        epoch: Optional[int] = None,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Compute IAAMSoftmax loss with AdaptiveDrop correction, filtering, and safeguard.

        Args:
            x: Input embeddings of shape (B, in_features).
            labels: Ground-truth speaker label indices of shape (B,).
            epoch: Optional explicit 1-indexed epoch override (defaults to self.current_epoch).
            return_diagnostics: If True, returns (loss, diagnostics_dict).

        Returns:
            loss: Scalar cross-entropy loss with angular margin penalty on retained samples,
                  or tuple (loss, diagnostics) if return_diagnostics=True.
        """
        # Validate inputs
        if not torch.isfinite(x).all():
            raise FloatingPointError("[NaN DETECTED] IAAMSoftmax input embeddings contain non-finite values.")

        B = x.size(0)
        curr_epoch = int(epoch) if epoch is not None else self.current_epoch

        # Edge case: empty batch
        if B == 0:
            dummy_loss = (x.sum() * 0.0) + (self.weight.sum() * 0.0)
            diag = {
                "num_corrected": 0,
                "num_filtered": 0,
                "num_retained": 0,
                "batch_size": 0,
                "keep_ratio": 1.0,
                "guard_activated": False,
                "loss": 0.0,
                "mean_target_cs": 0.0,
            }
            self.last_diagnostics = diag
            self.last_keep_mask = None
            return (dummy_loss, diag) if return_diagnostics else dummy_loss

        # Validate label ranges
        min_lbl = int(labels.min().item())
        max_lbl = int(labels.max().item())
        if min_lbl < 0 or max_lbl >= self.num_classes:
            raise ValueError(
                f"Labels out of range [0, {self.num_classes - 1}]: min={min_lbl}, max={max_lbl}"
            )

        # Enforce fp32 execution per Rule 5.1 and ADR-007
        x_fp32 = x.float()
        w_fp32 = self.weight.float()

        # Normalize feature embeddings and class weights to unit hypersphere
        x_norm = F.normalize(x_fp32, p=2, dim=1)
        w_norm = F.normalize(w_fp32, p=2, dim=1)

        # Cosine similarity matrix: cos(theta) = x_norm @ w_norm.T -> (B, num_classes)
        cos_theta = F.linear(x_norm, w_norm)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Active AdaptiveDrop decisions are computed under torch.no_grad()
        rectified_labels = labels.clone()
        num_corrected = 0
        guard_activated = False

        apply_adaptivedrop = self.training and self.use_adaptive_drop

        if apply_adaptivedrop:
            with torch.no_grad():
                # Extract cosine similarities to assigned classes: (B,)
                cs_assigned = cos_theta.gather(1, labels.unsqueeze(1)).squeeze(1)

                # -------------------------------------------------------------
                # Stage 1: Label Correction (epoch >= ESC)
                # Correct suspicious labels BEFORE filtering to prevent over-cleaning.
                # -------------------------------------------------------------
                if curr_epoch >= self.esc:
                    # Identify candidate classes excluding the currently assigned class
                    cos_theta_masked = cos_theta.clone()
                    cos_theta_masked.scatter_(1, labels.unsqueeze(1), -2.0)
                    best_cs_competing, best_competing_class = cos_theta_masked.max(dim=1)

                    # Condition: CS to assigned < tau AND max CS to competing >= tau_corr
                    corr_mask = (cs_assigned < self.tau) & (best_cs_competing >= self.tau_corr)
                    rectified_labels = torch.where(corr_mask, best_competing_class, labels)
                    num_corrected = int(corr_mask.sum().item())

                # -------------------------------------------------------------
                # Stage 2: Sample Filtering (epoch >= ESD)
                # Filter samples against their (potentially rectified) class center.
                # -------------------------------------------------------------
                if curr_epoch >= self.esd:
                    cs_target = cos_theta.gather(1, rectified_labels.unsqueeze(1)).squeeze(1)
                    keep_mask = cs_target >= self.tau
                    num_kept = int(keep_mask.sum().item())

                    # Minimum keep-ratio safeguard (ADR-001)
                    if self.min_keep_ratio > 0.0:
                        min_keep = max(1, math.ceil(B * self.min_keep_ratio))
                        if num_kept < min_keep:
                            guard_activated = True
                            # Deterministically retain top-k highest similarity samples
                            topk_idx = torch.topk(cs_target, k=min_keep, largest=True).indices
                            keep_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
                            keep_mask[topk_idx] = True
                            num_kept = min_keep
                            logger.info(
                                f"[AdaptiveDrop Guard] Activated at epoch {curr_epoch}: "
                                f"kept {num_kept}/{B} samples (min_keep_ratio={self.min_keep_ratio})"
                            )
                else:
                    keep_mask = torch.ones(B, dtype=torch.bool, device=x.device)
                    num_kept = B
        else:
            keep_mask = torch.ones(B, dtype=torch.bool, device=x.device)
            num_kept = B

        num_filtered = B - num_kept
        keep_ratio = float(num_kept) / float(B)

        # -------------------------------------------------------------
        # Stage 3: AAMSoftmax Loss Computation over Retained Samples
        # -------------------------------------------------------------
        if num_kept == 0:
            # Edge case: all samples dropped and safeguard disabled
            loss = (x.sum() * 0.0) + (self.weight.sum() * 0.0)
            mean_target_cs = 0.0
        else:
            cos_theta_selected = cos_theta[keep_mask]
            labels_selected = rectified_labels[keep_mask]

            # Angular margin computation on selected samples
            sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta_selected.pow(2), min=1e-7))
            phi = cos_theta_selected * self.cos_m - sin_theta * self.sin_m

            if self.easy_margin:
                phi = torch.where(cos_theta_selected > 0, phi, cos_theta_selected)
            else:
                phi = torch.where(cos_theta_selected > self.th, phi, cos_theta_selected - self.mm)

            one_hot = torch.zeros_like(cos_theta_selected)
            one_hot.scatter_(1, labels_selected.unsqueeze(1), 1.0)

            # ArcFace projection with scale s=30.0
            logits = (one_hot * phi) + ((1.0 - one_hot) * cos_theta_selected)
            logits = logits * self.scale

            loss = F.cross_entropy(logits, labels_selected, reduction=self.reduction)

            with torch.no_grad():
                target_cs = cos_theta_selected.gather(1, labels_selected.unsqueeze(1)).squeeze(1)
                mean_target_cs = float(target_cs.mean().item())

        # Record diagnostics
        loss_val = float(loss.item()) if loss.ndim == 0 else float(loss.mean().item())
        diag = {
            "num_corrected": num_corrected,
            "num_filtered": num_filtered,
            "num_retained": num_kept,
            "batch_size": B,
            "keep_ratio": round(keep_ratio, 4),
            "guard_activated": guard_activated,
            "loss": round(loss_val, 6),
            "mean_target_cs": round(mean_target_cs, 4),
            "epoch": curr_epoch,
        }
        self.last_diagnostics = diag
        self.last_keep_mask = keep_mask.detach()

        # Update epoch accumulators
        self._epoch_total_samples += B
        self._epoch_retained_samples += num_kept
        self._epoch_filtered_samples += num_filtered
        self._epoch_corrected_samples += num_corrected
        if guard_activated:
            self._epoch_guard_activations += 1
        self._epoch_batches += 1

        return (loss, diag) if return_diagnostics else loss


__all__ = ["IAAMSoftmax"]


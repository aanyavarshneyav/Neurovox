"""Evaluation metrics for speaker verification.

Provides threshold-free Equal Error Rate (EER), minimum Detection Cost Function
(minDCF at configurable P_target priors, normalized by default), DET curve
computation with endpoints, and speaker-level weighted bootstrap confidence
intervals across competing systems.

Adheres to NIST SRE conventions and matches speechbrain.utils.metric_stats.
"""

import logging
from typing import Any
import numpy as np
import torch

logger = logging.getLogger(__name__)


class MinDCFResult(dict):
    """Container for minDCF evaluation results supporting dict keys and attribute access."""

    def __init__(
        self,
        min_dcf: float,
        min_dcf_raw: float,
        threshold: float,
        p_target: float,
    ) -> None:
        super().__init__(
            min_dcf=min_dcf,
            min_dcf_raw=min_dcf_raw,
            threshold=threshold,
            p_target=p_target,
        )
        self.min_dcf = min_dcf
        self.min_dcf_raw = min_dcf_raw
        self.threshold = threshold
        self.p_target = p_target

    def __iter__(self):
        # Support both dictionary keys iteration and tuple unpacking (min_dcf, threshold)
        yield self.min_dcf
        yield self.threshold

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, int):
            return (self.min_dcf, self.threshold)[item]
        return super().__getitem__(item)


def _to_numpy_1d(
    arr: np.ndarray | torch.Tensor,
    name: str = "array",
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Safely convert tensor or array to 1D numpy array with type conversions."""
    if isinstance(arr, torch.Tensor):
        if arr.dtype in [torch.bfloat16, torch.float16]:
            arr = arr.float()
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=dtype).ravel()
    return arr


def _validate_inputs(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate shapes, absence of NaNs/Infs, binary labels {0, 1}, and class presence."""
    if len(scores) != len(labels):
        raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(labels)} labels.")
    if len(scores) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    if np.isnan(scores).any() or np.isinf(scores).any():
        raise ValueError("Scores array contains NaN or Inf values.")

    unique_labels = np.unique(labels)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(f"Labels must be binary in {{0, 1}}, found: {unique_labels.tolist()}")

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Both positive and negative samples required. Found {n_pos} pos, {n_neg} neg.")

    if weights is not None:
        weights = _to_numpy_1d(weights, name="weights", dtype=np.float64)
        if len(weights) != len(scores):
            raise ValueError(f"Length mismatch: {len(weights)} weights vs {len(scores)} scores.")
        if np.isnan(weights).any() or np.isinf(weights).any():
            raise ValueError("Weights array contains NaN or Inf values.")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")
        w_pos = float(np.sum(weights[labels == 1]))
        w_neg = float(np.sum(weights[labels == 0]))
        if w_pos <= 0 or w_neg <= 0:
            raise ValueError(f"Both classes must have positive total weight. Found {w_pos} pos, {w_neg} neg.")
    else:
        weights = np.ones_like(scores, dtype=np.float64)

    return scores, labels, weights


def _check_label_direction(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    strict: bool = False,
) -> None:
    """Check whether positive scores are systematically higher than negative scores."""
    mean_pos = float(np.mean(pos_scores))
    mean_neg = float(np.mean(neg_scores))
    if mean_pos < mean_neg:
        msg = (
            f"Target scores are systematically lower than impostor scores: "
            f"mean(target)={mean_pos:.4f} < mean(impostor)={mean_neg:.4f}. "
            f"Verify score polarity or label orientation."
        )
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)


def _compute_weighted_rates(
    pos_scores: np.ndarray,
    pos_weights: np.ndarray,
    neg_scores: np.ndarray,
    neg_weights: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted FRR and FAR across thresholds in O(N log N)."""
    w_pos_total = float(np.sum(pos_weights))
    w_neg_total = float(np.sum(neg_weights))

    # Sort positive scores and compute cumulative weights
    pos_order = np.argsort(pos_scores)
    sorted_pos = pos_scores[pos_order]
    cum_pos_w = np.cumsum(pos_weights[pos_order])

    idx_pos = np.searchsorted(sorted_pos, thresholds, side="right")
    pos_le_w = np.zeros_like(thresholds, dtype=np.float64)
    mask_pos = idx_pos > 0
    pos_le_w[mask_pos] = cum_pos_w[idx_pos[mask_pos] - 1]
    frr = pos_le_w / w_pos_total

    # Sort negative scores and compute cumulative weights
    neg_order = np.argsort(neg_scores)
    sorted_neg = neg_scores[neg_order]
    cum_neg_w = np.cumsum(neg_weights[neg_order])

    idx_neg = np.searchsorted(sorted_neg, thresholds, side="right")
    neg_le_w = np.zeros_like(thresholds, dtype=np.float64)
    mask_neg = idx_neg > 0
    neg_le_w[mask_neg] = cum_neg_w[idx_neg[mask_neg] - 1]
    far = (w_neg_total - neg_le_w) / w_neg_total

    return frr, far


def compute_eer(
    scores: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    strict: bool = False,
    weights: np.ndarray | torch.Tensor | None = None,
) -> tuple[float, float]:
    """Compute Equal Error Rate (EER) and its operating threshold.

    EER is the operating point where False Acceptance Rate (FAR / FPR) equals
    False Rejection Rate (FRR / FNR). Candidate thresholds include unique active
    score values and their intermediate midpoints.

    Args:
        scores: 1D array or tensor of similarity scores.
        labels: 1D array or tensor of binary ground-truth labels (1 for target, 0 for impostor).
        strict: If True, raises ValueError when mean(target) < mean(impostor).
            If False (e.g. during early training warmup), logs a loud warning.
        weights: Optional 1D array or tensor of non-negative per-trial weights.

    Returns:
        tuple of (eer, threshold):
            eer: Float in [0.0, 1.0] representing the fraction of errors at equal error point.
            threshold: Decision threshold minimizing absolute difference between FAR and FRR.
    """
    scores_np = _to_numpy_1d(scores, "scores", dtype=np.float64)
    labels_np = _to_numpy_1d(labels, "labels", dtype=np.int64)
    weights_np = _to_numpy_1d(weights, "weights", dtype=np.float64) if weights is not None else None

    scores_np, labels_np, weights_np = _validate_inputs(scores_np, labels_np, weights_np)

    pos_mask = labels_np == 1
    neg_mask = labels_np == 0

    pos_scores = scores_np[pos_mask]
    pos_weights = weights_np[pos_mask]
    neg_scores = scores_np[neg_mask]
    neg_weights = weights_np[neg_mask]

    _check_label_direction(pos_scores, neg_scores, strict=strict)

    # Candidate thresholds from scores with non-zero weight
    active_mask = weights_np > 0
    unique_scores = np.unique(scores_np[active_mask])
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    thresholds = np.sort(np.concatenate([unique_scores, midpoints]))

    frr, far = _compute_weighted_rates(pos_scores, pos_weights, neg_scores, neg_weights, thresholds)

    diff = np.abs(far - frr)
    min_idx = int(np.argmin(diff))

    eer = float((far[min_idx] + frr[min_idx]) / 2.0)
    best_threshold = float(thresholds[min_idx])

    return eer, best_threshold


def compute_mindcf(
    scores: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    normalize: bool = True,
    strict: bool = False,
    weights: np.ndarray | torch.Tensor | None = None,
) -> MinDCFResult:
    """Compute minimum Detection Cost Function (minDCF) and its optimal threshold.

    Evaluates NIST SRE detection cost function across all candidate thresholds:
        C_det = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
    where p_miss is FRR and p_fa is FAR.

    By default, minDCF is NORMALIZED by dividing by the default baseline cost:
        C_def = min(c_miss * p_target, c_fa * (1 - p_target))
        minDCF = C_det_min / C_def

    Returns a MinDCFResult dictionary exposing 'min_dcf' (normalized),
    'min_dcf_raw' (unnormalized), 'threshold', and 'p_target'.

    Args:
        scores: 1D array or tensor of real-valued similarity scores.
        labels: 1D array or tensor of binary ground-truth labels (1 for target, 0 for impostor).
        p_target: Prior probability of target trial (default 0.05 primary).
        c_miss: Cost of miss / false rejection (default 1.0).
        c_fa: Cost of false alarm / false acceptance (default 1.0).
        normalize: If True, min_dcf key is normalized by C_def (default True).
        strict: If True, raises ValueError when mean(target) < mean(impostor).
        weights: Optional 1D array or tensor of non-negative per-trial weights.

    Returns:
        MinDCFResult dictionary with keys 'min_dcf', 'min_dcf_raw', 'threshold', 'p_target'.
    """
    scores_np = _to_numpy_1d(scores, "scores", dtype=np.float64)
    labels_np = _to_numpy_1d(labels, "labels", dtype=np.int64)
    weights_np = _to_numpy_1d(weights, "weights", dtype=np.float64) if weights is not None else None

    scores_np, labels_np, weights_np = _validate_inputs(scores_np, labels_np, weights_np)

    pos_mask = labels_np == 1
    neg_mask = labels_np == 0

    pos_scores = scores_np[pos_mask]
    pos_weights = weights_np[pos_mask]
    neg_scores = scores_np[neg_mask]
    neg_weights = weights_np[neg_mask]

    _check_label_direction(pos_scores, neg_scores, strict=strict)

    active_mask = weights_np > 0
    unique_scores = np.unique(scores_np[active_mask])
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    thresholds = np.sort(np.concatenate([unique_scores, midpoints]))

    p_miss, p_fa = _compute_weighted_rates(pos_scores, pos_weights, neg_scores, neg_weights, thresholds)

    c_det = c_miss * p_miss * p_target + c_fa * p_fa * (1.0 - p_target)
    min_idx = int(np.argmin(c_det))
    raw_mindcf = float(c_det[min_idx])
    best_threshold = float(thresholds[min_idx])

    c_def = min(c_miss * p_target, c_fa * (1.0 - p_target))
    norm_mindcf = float(raw_mindcf / c_def)

    final_mindcf = norm_mindcf if normalize else raw_mindcf

    return MinDCFResult(
        min_dcf=final_mindcf,
        min_dcf_raw=raw_mindcf,
        threshold=best_threshold,
        p_target=p_target,
    )


def compute_det_curve(
    scores: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR) curve.

    Includes the accept-all (FAR=1, FRR=0) and reject-all (FAR=0, FRR=1) endpoints.

    Args:
        scores: 1D array or tensor of similarity scores.
        labels: 1D array or tensor of binary ground-truth labels.
        weights: Optional 1D array or tensor of non-negative per-trial weights.

    Returns:
        Dictionary containing 'far', 'frr', and 'thresholds'.
    """
    scores_np = _to_numpy_1d(scores, "scores", dtype=np.float64)
    labels_np = _to_numpy_1d(labels, "labels", dtype=np.int64)
    weights_np = _to_numpy_1d(weights, "weights", dtype=np.float64) if weights is not None else None

    scores_np, labels_np, weights_np = _validate_inputs(scores_np, labels_np, weights_np)

    pos_mask = labels_np == 1
    neg_mask = labels_np == 0

    pos_scores = scores_np[pos_mask]
    pos_weights = weights_np[pos_mask]
    neg_scores = scores_np[neg_mask]
    neg_weights = weights_np[neg_mask]

    active_mask = weights_np > 0
    unique_scores = np.unique(scores_np[active_mask])
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0

    # Add endpoints: accept-all (threshold below min score) and reject-all (threshold above max score)
    min_score = float(np.min(unique_scores))
    max_score = float(np.max(unique_scores))
    endpoint_low = min_score - 1.0
    endpoint_high = max_score + 1.0

    thresholds = np.sort(np.concatenate([[endpoint_low], unique_scores, midpoints, [endpoint_high]]))

    frr, far = _compute_weighted_rates(pos_scores, pos_weights, neg_scores, neg_weights, thresholds)

    return {
        "far": far,
        "frr": frr,
        "thresholds": thresholds,
    }


def speaker_bootstrap_ci(
    scores_a: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    spk_a: np.ndarray | list[Any],
    spk_b: np.ndarray | list[Any],
    scores_b: np.ndarray | torch.Tensor | None = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    p_target_primary: float = 0.05,
    p_target_secondary: float = 0.01,
) -> dict[str, Any]:
    """Compute SPEAKER-LEVEL clustered bootstrap confidence intervals.

    Resamples unique speakers with replacement. For each bootstrap iteration:
        - weight = count(spk_a) for target trials (spk_a == spk_b)
        - weight = count(spk_a) * count(spk_b) for impostor trials (spk_a != spk_b)

    Evaluates both systems under identical bootstrap weights, yielding valid CIs
    for System A, System B, and the paired differences (A - B).

    Args:
        scores_a: Scores for System A.
        labels: Binary labels (1 target, 0 impostor).
        spk_a: Array of speaker IDs for the first utterance in each trial.
        spk_b: Array of speaker IDs for the second utterance in each trial.
        scores_b: Optional scores for System B for paired comparison.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: Desired coverage probability (default 0.95).
        seed: Random seed for reproducible resampling.
        p_target_primary: Primary minDCF prior (default 0.05).
        p_target_secondary: Secondary indicative minDCF prior (default 0.01).

    Returns:
        Comprehensive dictionary of point estimates, bootstrap means, 95% CIs,
        and difference statistics.
    """
    scores_a_np = _to_numpy_1d(scores_a, "scores_a", dtype=np.float64)
    labels_np = _to_numpy_1d(labels, "labels", dtype=np.int64)
    scores_a_np, labels_np, _ = _validate_inputs(scores_a_np, labels_np)

    spk_a_np = np.asarray(spk_a).astype(str).ravel()
    spk_b_np = np.asarray(spk_b).astype(str).ravel()
    if len(spk_a_np) != len(scores_a_np) or len(spk_b_np) != len(scores_a_np):
        raise ValueError("Length mismatch between scores and speaker arrays.")

    scores_b_np: np.ndarray | None = None
    if scores_b is not None:
        scores_b_np = _to_numpy_1d(scores_b, "scores_b", dtype=np.float64)
        if len(scores_b_np) != len(scores_a_np):
            raise ValueError("Length mismatch between scores_a and scores_b.")

    # Unique speakers participating in trials
    unique_speakers = np.unique(np.concatenate([spk_a_np, spk_b_np]))
    n_spks = len(unique_speakers)

    # Point estimates on unweighted full dataset
    pt_eer_a, _ = compute_eer(scores_a_np, labels_np, strict=False)
    pt_dcf05_a = compute_mindcf(scores_a_np, labels_np, p_target=p_target_primary, normalize=True)
    pt_dcf01_a = compute_mindcf(scores_a_np, labels_np, p_target=p_target_secondary, normalize=True)

    pt_res_a = {
        "eer": pt_eer_a,
        "mindcf_p05": pt_dcf05_a["min_dcf"],
        "mindcf_p05_raw": pt_dcf05_a["min_dcf_raw"],
        "mindcf_p01": pt_dcf01_a["min_dcf"],
        "mindcf_p01_raw": pt_dcf01_a["min_dcf_raw"],
    }

    pt_res_b: dict[str, float] | None = None
    pt_diff: dict[str, float] | None = None
    if scores_b_np is not None:
        pt_eer_b, _ = compute_eer(scores_b_np, labels_np, strict=False)
        pt_dcf05_b = compute_mindcf(scores_b_np, labels_np, p_target=p_target_primary, normalize=True)
        pt_dcf01_b = compute_mindcf(scores_b_np, labels_np, p_target=p_target_secondary, normalize=True)
        pt_res_b = {
            "eer": pt_eer_b,
            "mindcf_p05": pt_dcf05_b["min_dcf"],
            "mindcf_p05_raw": pt_dcf05_b["min_dcf_raw"],
            "mindcf_p01": pt_dcf01_b["min_dcf"],
            "mindcf_p01_raw": pt_dcf01_b["min_dcf_raw"],
        }
        pt_diff = {
            "diff_eer": pt_eer_a - pt_eer_b,
            "diff_mindcf_p05": pt_dcf05_a["min_dcf"] - pt_dcf05_b["min_dcf"],
            "diff_mindcf_p01": pt_dcf01_a["min_dcf"] - pt_dcf01_b["min_dcf"],
        }

    rng = np.random.default_rng(seed)

    eer_a_samples: list[float] = []
    dcf05_a_samples: list[float] = []
    dcf01_a_samples: list[float] = []

    eer_b_samples: list[float] = []
    dcf05_b_samples: list[float] = []
    dcf01_b_samples: list[float] = []

    diff_eer_samples: list[float] = []
    diff_dcf05_samples: list[float] = []
    diff_dcf01_samples: list[float] = []

    n_skipped = 0

    # Pre-map speaker strings to indices 0..n_spks-1 for fast lookup
    spk2idx = {s: i for i, s in enumerate(unique_speakers)}
    idx_a = np.array([spk2idx[s] for s in spk_a_np], dtype=np.int32)
    idx_b = np.array([spk2idx[s] for s in spk_b_np], dtype=np.int32)
    is_target = labels_np == 1

    for _ in range(n_bootstrap):
        sampled_spk_indices = rng.integers(0, n_spks, size=n_spks)
        counts = np.bincount(sampled_spk_indices, minlength=n_spks)

        count_a = counts[idx_a]
        count_b = counts[idx_b]

        weights = np.empty_like(count_a, dtype=np.float64)
        weights[is_target] = count_a[is_target]
        weights[~is_target] = count_a[~is_target] * count_b[~is_target]

        w_pos = float(np.sum(weights[is_target]))
        w_neg = float(np.sum(weights[~is_target]))
        if w_pos <= 0 or w_neg <= 0:
            n_skipped += 1
            continue

        eer_a, _ = compute_eer(scores_a_np, labels_np, strict=False, weights=weights)
        dcf05_a = compute_mindcf(scores_a_np, labels_np, p_target=p_target_primary, normalize=True, strict=False, weights=weights)
        dcf01_a = compute_mindcf(scores_a_np, labels_np, p_target=p_target_secondary, normalize=True, strict=False, weights=weights)

        eer_a_samples.append(eer_a)
        dcf05_a_samples.append(dcf05_a["min_dcf"])
        dcf01_a_samples.append(dcf01_a["min_dcf"])

        if scores_b_np is not None:
            eer_b, _ = compute_eer(scores_b_np, labels_np, strict=False, weights=weights)
            dcf05_b = compute_mindcf(scores_b_np, labels_np, p_target=p_target_primary, normalize=True, strict=False, weights=weights)
            dcf01_b = compute_mindcf(scores_b_np, labels_np, p_target=p_target_secondary, normalize=True, strict=False, weights=weights)

            eer_b_samples.append(eer_b)
            dcf05_b_samples.append(dcf05_b["min_dcf"])
            dcf01_b_samples.append(dcf01_b["min_dcf"])

            diff_eer_samples.append(eer_a - eer_b)
            diff_dcf05_samples.append(dcf05_a["min_dcf"] - dcf05_b["min_dcf"])
            diff_dcf01_samples.append(dcf01_a["min_dcf"] - dcf01_b["min_dcf"])

    alpha = 1.0 - confidence_level
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)

    results: dict[str, Any] = {
        "point_estimate_a": pt_res_a,
        "bootstrap_mean_a": {
            "eer": float(np.mean(eer_a_samples)),
            "mindcf_p05": float(np.mean(dcf05_a_samples)),
            "mindcf_p01": float(np.mean(dcf01_a_samples)),
        },
        "ci_95_a": {
            "eer": (float(np.percentile(eer_a_samples, low_pct)), float(np.percentile(eer_a_samples, high_pct))),
            "mindcf_p05": (float(np.percentile(dcf05_a_samples, low_pct)), float(np.percentile(dcf05_a_samples, high_pct))),
            "mindcf_p01": (float(np.percentile(dcf01_a_samples, low_pct)), float(np.percentile(dcf01_a_samples, high_pct))),
        },
        "n_valid_bootstrap": len(eer_a_samples),
        "n_skipped_degenerate": n_skipped,
    }

    if scores_b_np is not None:
        results.update(
            {
                "point_estimate_b": pt_res_b,
                "bootstrap_mean_b": {
                    "eer": float(np.mean(eer_b_samples)),
                    "mindcf_p05": float(np.mean(dcf05_b_samples)),
                    "mindcf_p01": float(np.mean(dcf01_b_samples)),
                },
                "ci_95_b": {
                    "eer": (float(np.percentile(eer_b_samples, low_pct)), float(np.percentile(eer_b_samples, high_pct))),
                    "mindcf_p05": (float(np.percentile(dcf05_b_samples, low_pct)), float(np.percentile(dcf05_b_samples, high_pct))),
                    "mindcf_p01": (float(np.percentile(dcf01_b_samples, low_pct)), float(np.percentile(dcf01_b_samples, high_pct))),
                },
                "diff_point_estimate": pt_diff,
                "diff_bootstrap_mean": {
                    "diff_eer": float(np.mean(diff_eer_samples)),
                    "diff_mindcf_p05": float(np.mean(diff_dcf05_samples)),
                    "diff_mindcf_p01": float(np.mean(diff_dcf01_samples)),
                },
                "diff_ci_95": {
                    "diff_eer": (float(np.percentile(diff_eer_samples, low_pct)), float(np.percentile(diff_eer_samples, high_pct))),
                    "diff_mindcf_p05": (float(np.percentile(diff_dcf05_samples, low_pct)), float(np.percentile(diff_dcf05_samples, high_pct))),
                    "diff_mindcf_p01": (float(np.percentile(diff_dcf01_samples, low_pct)), float(np.percentile(diff_dcf01_samples, high_pct))),
                },
            }
        )

    return results


# Retain alias paired_bootstrap_ci for compatibility
paired_bootstrap_ci = speaker_bootstrap_ci


def speaker_bootstrap_two_sample_difference(
    scores_1: np.ndarray | torch.Tensor,
    labels_1: np.ndarray | torch.Tensor,
    spk_a_1: np.ndarray | list[Any],
    spk_b_1: np.ndarray | list[Any],
    scores_2: np.ndarray | torch.Tensor,
    labels_2: np.ndarray | torch.Tensor,
    spk_a_2: np.ndarray | list[Any],
    spk_b_2: np.ndarray | list[Any],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    p_target_primary: float = 0.05,
    p_target_secondary: float = 0.01,
) -> dict[str, Any]:
    """Compute independent SPEAKER-LEVEL bootstrap CIs for two disjoint sets and their difference (Set 2 - Set 1).

    Used when comparing performance across two independent evaluation partitions (e.g.,
    test-other minus test-clean) that do not share trials or speakers.

    Args:
        scores_1: Verification scores on partition 1 (e.g., test-clean).
        labels_1: Verification labels on partition 1.
        spk_a_1: Speaker IDs for utterance A in partition 1.
        spk_b_1: Speaker IDs for utterance B in partition 1.
        scores_2: Verification scores on partition 2 (e.g., test-other).
        labels_2: Verification labels on partition 2.
        spk_a_2: Speaker IDs for utterance A in partition 2.
        spk_b_2: Speaker IDs for utterance B in partition 2.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: Coverage probability (default 0.95).
        seed: Random seed for reproducibility.
        p_target_primary: Primary minDCF prior (default 0.05).
        p_target_secondary: Secondary indicative minDCF prior (default 0.01).

    Returns:
        Comprehensive dictionary of point estimates, bootstrap means, 95% CIs for each set,
        and their difference (Set 2 minus Set 1).
    """
    scores_1_np = _to_numpy_1d(scores_1, "scores_1", dtype=np.float64)
    labels_1_np = _to_numpy_1d(labels_1, "labels_1", dtype=np.int64)
    scores_1_np, labels_1_np, _ = _validate_inputs(scores_1_np, labels_1_np)

    scores_2_np = _to_numpy_1d(scores_2, "scores_2", dtype=np.float64)
    labels_2_np = _to_numpy_1d(labels_2, "labels_2", dtype=np.int64)
    scores_2_np, labels_2_np, _ = _validate_inputs(scores_2_np, labels_2_np)

    spk_a_1_np = np.asarray(spk_a_1).astype(str).ravel()
    spk_b_1_np = np.asarray(spk_b_1).astype(str).ravel()
    spk_a_2_np = np.asarray(spk_a_2).astype(str).ravel()
    spk_b_2_np = np.asarray(spk_b_2).astype(str).ravel()

    if len(spk_a_1_np) != len(scores_1_np) or len(spk_b_1_np) != len(scores_1_np):
        raise ValueError("Length mismatch in partition 1 inputs.")
    if len(spk_a_2_np) != len(scores_2_np) or len(spk_b_2_np) != len(scores_2_np):
        raise ValueError("Length mismatch in partition 2 inputs.")

    # Unique speakers per partition
    spks_1 = np.unique(np.concatenate([spk_a_1_np, spk_b_1_np]))
    spks_2 = np.unique(np.concatenate([spk_a_2_np, spk_b_2_np]))
    n_spks_1 = len(spks_1)
    n_spks_2 = len(spks_2)

    # Point estimates
    pt_eer_1, _ = compute_eer(scores_1_np, labels_1_np, strict=False)
    pt_dcf05_1 = compute_mindcf(scores_1_np, labels_1_np, p_target=p_target_primary, normalize=True)
    pt_dcf01_1 = compute_mindcf(scores_1_np, labels_1_np, p_target=p_target_secondary, normalize=True)

    pt_eer_2, _ = compute_eer(scores_2_np, labels_2_np, strict=False)
    pt_dcf05_2 = compute_mindcf(scores_2_np, labels_2_np, p_target=p_target_primary, normalize=True)
    pt_dcf01_2 = compute_mindcf(scores_2_np, labels_2_np, p_target=p_target_secondary, normalize=True)

    pt_res_1 = {
        "eer": pt_eer_1,
        "mindcf_p05": pt_dcf05_1["min_dcf"],
        "mindcf_p05_raw": pt_dcf05_1["min_dcf_raw"],
        "mindcf_p01": pt_dcf01_1["min_dcf"],
        "mindcf_p01_raw": pt_dcf01_1["min_dcf_raw"],
    }
    pt_res_2 = {
        "eer": pt_eer_2,
        "mindcf_p05": pt_dcf05_2["min_dcf"],
        "mindcf_p05_raw": pt_dcf05_2["min_dcf_raw"],
        "mindcf_p01": pt_dcf01_2["min_dcf"],
        "mindcf_p01_raw": pt_dcf01_2["min_dcf_raw"],
    }
    pt_diff = {
        "diff_eer": pt_eer_2 - pt_eer_1,
        "diff_mindcf_p05": pt_dcf05_2["min_dcf"] - pt_dcf05_1["min_dcf"],
        "diff_mindcf_p01": pt_dcf01_2["min_dcf"] - pt_dcf01_1["min_dcf"],
    }

    spk2idx_1 = {s: i for i, s in enumerate(spks_1)}
    spk2idx_2 = {s: i for i, s in enumerate(spks_2)}
    idx_a_1 = np.array([spk2idx_1[s] for s in spk_a_1_np], dtype=np.int32)
    idx_b_1 = np.array([spk2idx_1[s] for s in spk_b_1_np], dtype=np.int32)
    idx_a_2 = np.array([spk2idx_2[s] for s in spk_a_2_np], dtype=np.int32)
    idx_b_2 = np.array([spk2idx_2[s] for s in spk_b_2_np], dtype=np.int32)

    is_target_1 = labels_1_np == 1
    is_target_2 = labels_2_np == 1

    rng = np.random.default_rng(seed)

    eer_1_samples, dcf05_1_samples, dcf01_1_samples = [], [], []
    eer_2_samples, dcf05_2_samples, dcf01_2_samples = [], [], []
    diff_eer_samples, diff_dcf05_samples, diff_dcf01_samples = [], [], []
    n_skipped = 0

    for _ in range(n_bootstrap):
        # Sample partition 1
        s_idx_1 = rng.integers(0, n_spks_1, size=n_spks_1)
        cnt_1 = np.bincount(s_idx_1, minlength=n_spks_1)
        w_1 = np.empty(len(scores_1_np), dtype=np.float64)
        w_1[is_target_1] = cnt_1[idx_a_1[is_target_1]]
        w_1[~is_target_1] = cnt_1[idx_a_1[~is_target_1]] * cnt_1[idx_b_1[~is_target_1]]

        # Sample partition 2
        s_idx_2 = rng.integers(0, n_spks_2, size=n_spks_2)
        cnt_2 = np.bincount(s_idx_2, minlength=n_spks_2)
        w_2 = np.empty(len(scores_2_np), dtype=np.float64)
        w_2[is_target_2] = cnt_2[idx_a_2[is_target_2]]
        w_2[~is_target_2] = cnt_2[idx_a_2[~is_target_2]] * cnt_2[idx_b_2[~is_target_2]]

        if np.sum(w_1[is_target_1]) <= 0 or np.sum(w_1[~is_target_1]) <= 0 or \
           np.sum(w_2[is_target_2]) <= 0 or np.sum(w_2[~is_target_2]) <= 0:
            n_skipped += 1
            continue

        e1, _ = compute_eer(scores_1_np, labels_1_np, strict=False, weights=w_1)
        d05_1 = compute_mindcf(scores_1_np, labels_1_np, p_target=p_target_primary, normalize=True, strict=False, weights=w_1)
        d01_1 = compute_mindcf(scores_1_np, labels_1_np, p_target=p_target_secondary, normalize=True, strict=False, weights=w_1)

        e2, _ = compute_eer(scores_2_np, labels_2_np, strict=False, weights=w_2)
        d05_2 = compute_mindcf(scores_2_np, labels_2_np, p_target=p_target_primary, normalize=True, strict=False, weights=w_2)
        d01_2 = compute_mindcf(scores_2_np, labels_2_np, p_target=p_target_secondary, normalize=True, strict=False, weights=w_2)

        eer_1_samples.append(e1)
        dcf05_1_samples.append(d05_1["min_dcf"])
        dcf01_1_samples.append(d01_1["min_dcf"])

        eer_2_samples.append(e2)
        dcf05_2_samples.append(d05_2["min_dcf"])
        dcf01_2_samples.append(d01_2["min_dcf"])

        diff_eer_samples.append(e2 - e1)
        diff_dcf05_samples.append(d05_2["min_dcf"] - d05_1["min_dcf"])
        diff_dcf01_samples.append(d01_2["min_dcf"] - d01_1["min_dcf"])

    alpha = 1.0 - confidence_level
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)

    return {
        "point_estimate_1": pt_res_1,
        "bootstrap_mean_1": {
            "eer": float(np.mean(eer_1_samples)),
            "mindcf_p05": float(np.mean(dcf05_1_samples)),
            "mindcf_p01": float(np.mean(dcf01_1_samples)),
        },
        "ci_95_1": {
            "eer": (float(np.percentile(eer_1_samples, low_pct)), float(np.percentile(eer_1_samples, high_pct))),
            "mindcf_p05": (float(np.percentile(dcf05_1_samples, low_pct)), float(np.percentile(dcf05_1_samples, high_pct))),
            "mindcf_p01": (float(np.percentile(dcf01_1_samples, low_pct)), float(np.percentile(dcf01_1_samples, high_pct))),
        },
        "point_estimate_2": pt_res_2,
        "bootstrap_mean_2": {
            "eer": float(np.mean(eer_2_samples)),
            "mindcf_p05": float(np.mean(dcf05_2_samples)),
            "mindcf_p01": float(np.mean(dcf01_2_samples)),
        },
        "ci_95_2": {
            "eer": (float(np.percentile(eer_2_samples, low_pct)), float(np.percentile(eer_2_samples, high_pct))),
            "mindcf_p05": (float(np.percentile(dcf05_2_samples, low_pct)), float(np.percentile(dcf05_2_samples, high_pct))),
            "mindcf_p01": (float(np.percentile(dcf01_2_samples, low_pct)), float(np.percentile(dcf01_2_samples, high_pct))),
        },
        "diff_point_estimate": pt_diff,
        "diff_bootstrap_mean": {
            "diff_eer": float(np.mean(diff_eer_samples)),
            "diff_mindcf_p05": float(np.mean(diff_dcf05_samples)),
            "diff_mindcf_p01": float(np.mean(diff_dcf01_samples)),
        },
        "diff_ci_95": {
            "diff_eer": (float(np.percentile(diff_eer_samples, low_pct)), float(np.percentile(diff_eer_samples, high_pct))),
            "diff_mindcf_p05": (float(np.percentile(diff_dcf05_samples, low_pct)), float(np.percentile(diff_dcf05_samples, high_pct))),
            "diff_mindcf_p01": (float(np.percentile(diff_dcf01_samples, low_pct)), float(np.percentile(diff_dcf01_samples, high_pct))),
        },
        "n_valid_bootstrap": len(eer_1_samples),
        "n_skipped_degenerate": n_skipped,
    }

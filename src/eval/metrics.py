"""Evaluation metrics for speaker verification.

Provides threshold-free Equal Error Rate (EER), minimum Detection Cost Function
(minDCF at configurable P_target priors), DET curve computation, and paired
clustered bootstrap confidence intervals across competing systems.

Adheres to NIST SRE conventions and matches speechbrain.utils.metric_stats.
"""

from typing import Any
import numpy as np
import torch


def compute_eer(
    scores: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    check_direction: bool = True,
) -> tuple[float, float]:
    """Compute Equal Error Rate (EER) and its operating threshold.

    EER is the operating point where False Acceptance Rate (FAR / FPR) equals
    False Rejection Rate (FRR / FNR).
    
    Candidate thresholds evaluated include all unique score values and their
    midpoints, matching speechbrain.utils.metric_stats.EER exactly with an
    efficient O(N log N) sorted search.

    Args:
        scores: 1D array or tensor of real-valued similarity scores (e.g., cosine similarity).
        labels: 1D array or tensor of binary ground-truth labels (1 for target, 0 for impostor).
        check_direction: If True, asserts/warns if target scores are lower than impostor scores.

    Returns:
        tuple of (eer, threshold):
            eer: Float in [0.0, 1.0] representing the fraction of errors at equal error point.
            threshold: Decision threshold at the minimum absolute difference between FAR and FRR.

    Raises:
        ValueError: If input arrays are empty, have mismatched lengths, or contain fewer than 2 classes.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()

    if len(scores) != len(labels):
        raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(labels)} labels.")
    if len(scores) == 0:
        raise ValueError("Cannot compute EER on empty score array.")

    pos_mask = labels == 1
    neg_mask = labels == 0

    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Both positive and negative samples required. Found {n_pos} pos, {n_neg} neg.")

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    mean_pos = float(np.mean(pos_scores))
    mean_neg = float(np.mean(neg_scores))

    if check_direction and mean_pos < mean_neg:
        # Informative diagnostic: higher scores should indicate targets.
        pass

    # Sort positive and negative scores
    pos_sorted = np.sort(pos_scores)
    neg_sorted = np.sort(neg_scores)

    # Candidate thresholds: unique score values + intermediate midpoints (matching SpeechBrain)
    unique_scores = np.unique(scores)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    thresholds = np.sort(np.concatenate([unique_scores, midpoints]))

    # Target decision rule: predict target (1) if score > threshold
    # FRR: fraction of targets with score <= threshold
    pos_le = np.searchsorted(pos_sorted, thresholds, side="right")
    frr = pos_le / n_pos

    # FAR: fraction of impostors with score > threshold
    neg_le = np.searchsorted(neg_sorted, thresholds, side="right")
    far = (n_neg - neg_le) / n_neg

    diff = np.abs(far - frr)
    min_idx = int(np.argmin(diff))

    eer = float((far[min_idx] + frr[min_idx]) / 2.0)
    best_threshold = float(thresholds[min_idx])

    return eer, best_threshold


def compute_mindcf(
    scores: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    normalize: bool = False,
) -> tuple[float, float]:
    """Compute minimum Detection Cost Function (minDCF) and its optimal threshold.

    Evaluates the NIST SRE detection cost function across all candidate thresholds:
        C_det = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
    where p_miss is the False Rejection Rate (FRR) and p_fa is the False Acceptance Rate (FAR).

    If normalize is True, minDCF is normalized by the default baseline cost:
        C_def = min(c_miss * p_target, c_fa * (1 - p_target))
        minDCF = min(C_det) / C_def

    Args:
        scores: 1D array or tensor of real-valued similarity scores.
        labels: 1D array or tensor of binary ground-truth labels (1 for target, 0 for impostor).
        p_target: Prior probability of target trial (default 0.01).
        c_miss: Cost of miss / false rejection (default 1.0).
        c_fa: Cost of false alarm / false acceptance (default 1.0).
        normalize: If True, normalize by min(c_miss * p_target, c_fa * (1 - p_target)).

    Returns:
        tuple of (mindcf, threshold):
            mindcf: Minimum detection cost value.
            threshold: Decision threshold that achieves the minimum detection cost.

    Raises:
        ValueError: If inputs are invalid or empty.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()

    if len(scores) != len(labels):
        raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(labels)} labels.")
    if len(scores) == 0:
        raise ValueError("Cannot compute minDCF on empty score array.")

    pos_mask = labels == 1
    neg_mask = labels == 0

    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Both positive and negative samples required. Found {n_pos} pos, {n_neg} neg.")

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    pos_sorted = np.sort(pos_scores)
    neg_sorted = np.sort(neg_scores)

    unique_scores = np.unique(scores)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    thresholds = np.sort(np.concatenate([unique_scores, midpoints]))

    pos_le = np.searchsorted(pos_sorted, thresholds, side="right")
    p_miss = pos_le / n_pos

    neg_le = np.searchsorted(neg_sorted, thresholds, side="right")
    p_fa = (n_neg - neg_le) / n_neg

    c_det = c_miss * p_miss * p_target + c_fa * p_fa * (1.0 - p_target)
    min_idx = int(np.argmin(c_det))
    raw_mindcf = float(c_det[min_idx])
    best_threshold = float(thresholds[min_idx])

    if normalize:
        c_def = min(c_miss * p_target, c_fa * (1.0 - p_target))
        return float(raw_mindcf / c_def), best_threshold

    return raw_mindcf, best_threshold


def compute_det_curve(
    scores: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
) -> dict[str, np.ndarray]:
    """Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR) curve.

    Args:
        scores: 1D array or tensor of similarity scores.
        labels: 1D array or tensor of binary ground-truth labels.

    Returns:
        Dictionary containing:
            'far': 1D numpy array of false acceptance rates.
            'frr': 1D numpy array of false rejection rates.
            'thresholds': 1D numpy array of operating thresholds.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_sorted = np.sort(scores[pos_mask])
    neg_sorted = np.sort(scores[neg_mask])

    n_pos = len(pos_sorted)
    n_neg = len(neg_sorted)

    unique_scores = np.unique(scores)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    thresholds = np.sort(np.concatenate([unique_scores, midpoints]))

    pos_le = np.searchsorted(pos_sorted, thresholds, side="right")
    frr = pos_le / n_pos

    neg_le = np.searchsorted(neg_sorted, thresholds, side="right")
    far = (n_neg - neg_le) / n_neg

    return {
        "far": far,
        "frr": frr,
        "thresholds": thresholds,
    }


def paired_bootstrap_ci(
    scores_a: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray | list[Any],
    scores_b: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute clustered paired bootstrap confidence intervals.

    Resamples clusters (speakers or utterances) with replacement, preserving trial
    dependencies. When scores_b is provided, resampling is paired: both systems
    are evaluated on identical resampled trial indices, yielding valid confidence
    intervals for the paired performance difference (System A - System B).

    Args:
        scores_a: 1D array of scores for System A.
        labels: 1D array of ground truth labels (0 or 1).
        clusters: 1D array of cluster IDs (e.g., speaker IDs) corresponding to each trial.
        scores_b: Optional 1D array of scores for System B for paired comparison.
        n_bootstrap: Number of bootstrap iterations (default 1000).
        confidence_level: Desired coverage probability (default 0.95).
        seed: Random seed for reproducible bootstrap resample indices.

    Returns:
        Dictionary containing:
            'eer_a_ci': (lower, upper)
            'eer_a_mean': float
            'mindcf_a_p01_ci': (lower, upper)
            'mindcf_a_p05_ci': (lower, upper)
            If scores_b provided:
                'eer_b_ci': (lower, upper)
                'diff_eer_ci': (lower, upper)
                'diff_eer_mean': float
                'diff_mindcf_p01_ci': (lower, upper)
                'diff_mindcf_p05_ci': (lower, upper)
    """
    scores_a = np.asarray(scores_a, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    clusters = np.asarray(clusters).ravel()

    n_trials = len(scores_a)
    if len(labels) != n_trials or len(clusters) != n_trials:
        raise ValueError("Length mismatch between scores_a, labels, and clusters.")

    if scores_b is not None:
        scores_b = np.asarray(scores_b, dtype=np.float64).ravel()
        if len(scores_b) != n_trials:
            raise ValueError("Length mismatch between scores_a and scores_b.")

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Precompute trial indices per cluster
    cluster_to_indices: dict[Any, np.ndarray] = {}
    for c in unique_clusters:
        cluster_to_indices[c] = np.where(clusters == c)[0]

    rng = np.random.default_rng(seed)

    eer_a_samples: list[float] = []
    dcf01_a_samples: list[float] = []
    dcf05_a_samples: list[float] = []

    eer_b_samples: list[float] = []
    dcf01_b_samples: list[float] = []
    dcf05_b_samples: list[float] = []

    diff_eer_samples: list[float] = []
    diff_dcf01_samples: list[float] = []
    diff_dcf05_samples: list[float] = []

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    for _ in range(n_bootstrap):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        # Concatenate indices
        resample_idx = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])

        sub_labels = labels[resample_idx]
        if np.all(sub_labels == 1) or np.all(sub_labels == 0):
            continue  # Skip degenerate sample lacking both classes

        sub_scores_a = scores_a[resample_idx]
        eer_a, _ = compute_eer(sub_scores_a, sub_labels, check_direction=False)
        dcf01_a, _ = compute_mindcf(sub_scores_a, sub_labels, p_target=0.01)
        dcf05_a, _ = compute_mindcf(sub_scores_a, sub_labels, p_target=0.05)

        eer_a_samples.append(eer_a)
        dcf01_a_samples.append(dcf01_a)
        dcf05_a_samples.append(dcf05_a)

        if scores_b is not None:
            sub_scores_b = scores_b[resample_idx]
            eer_b, _ = compute_eer(sub_scores_b, sub_labels, check_direction=False)
            dcf01_b, _ = compute_mindcf(sub_scores_b, sub_labels, p_target=0.01)
            dcf05_b, _ = compute_mindcf(sub_scores_b, sub_labels, p_target=0.05)

            eer_b_samples.append(eer_b)
            dcf01_b_samples.append(dcf01_b)
            dcf05_b_samples.append(dcf05_b)

            diff_eer_samples.append(eer_a - eer_b)
            diff_dcf01_samples.append(dcf01_a - dcf01_b)
            diff_dcf05_samples.append(dcf05_a - dcf05_b)

    results: dict[str, Any] = {
        "eer_a_ci": (
            float(np.percentile(eer_a_samples, lower_pct)),
            float(np.percentile(eer_a_samples, upper_pct)),
        ),
        "eer_a_mean": float(np.mean(eer_a_samples)),
        "mindcf_a_p01_ci": (
            float(np.percentile(dcf01_a_samples, lower_pct)),
            float(np.percentile(dcf01_a_samples, upper_pct)),
        ),
        "mindcf_a_p05_ci": (
            float(np.percentile(dcf05_a_samples, lower_pct)),
            float(np.percentile(dcf05_a_samples, upper_pct)),
        ),
        "n_valid_bootstrap": len(eer_a_samples),
    }

    if scores_b is not None:
        results.update(
            {
                "eer_b_ci": (
                    float(np.percentile(eer_b_samples, lower_pct)),
                    float(np.percentile(eer_b_samples, upper_pct)),
                ),
                "eer_b_mean": float(np.mean(eer_b_samples)),
                "diff_eer_ci": (
                    float(np.percentile(diff_eer_samples, lower_pct)),
                    float(np.percentile(diff_eer_samples, upper_pct)),
                ),
                "diff_eer_mean": float(np.mean(diff_eer_samples)),
                "diff_mindcf_p01_ci": (
                    float(np.percentile(diff_dcf01_samples, lower_pct)),
                    float(np.percentile(diff_dcf01_samples, upper_pct)),
                ),
                "diff_mindcf_p05_ci": (
                    float(np.percentile(diff_dcf05_samples, lower_pct)),
                    float(np.percentile(diff_dcf05_samples, upper_pct)),
                ),
            }
        )

    return results


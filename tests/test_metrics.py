"""Unit tests for speaker verification metrics (EER, minDCF, DET, and paired bootstrap).

Verifies numerical correctness, edge cases, label orientation behavior,
analytic asymptotic limits, and parity against speechbrain.utils.metric_stats.
"""

import math
import numpy as np
import pytest
import scipy.special
import torch

from src.eval.metrics import (
    compute_eer,
    compute_mindcf,
    compute_det_curve,
    paired_bootstrap_ci,
)

try:
    import speechbrain.utils.metric_stats as sb_stats
    HAS_SPEECHBRAIN = True
except ImportError:
    HAS_SPEECHBRAIN = False


def test_perfect_separation():
    """Verify EER and minDCF are exactly 0.0 when scores are perfectly separable."""
    pos_scores = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    neg_scores = np.array([-6.0, -5.0, -4.0, -3.0, -2.0])

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])

    eer, threshold = compute_eer(scores, labels)
    assert eer == 0.0, f"Expected EER 0.0, got {eer}"
    # Threshold should sit between max negative (-2.0) and min positive (2.0)
    assert -2.0 <= threshold <= 2.0

    dcf01, _ = compute_mindcf(scores, labels, p_target=0.01)
    dcf05, _ = compute_mindcf(scores, labels, p_target=0.05)
    assert dcf01 == 0.0, f"Expected minDCF(0.01) 0.0, got {dcf01}"
    assert dcf05 == 0.0, f"Expected minDCF(0.05) 0.0, got {dcf05}"


def test_random_scores():
    """Verify EER is approximately 50% when positive and negative scores are identical."""
    rng = np.random.default_rng(123)
    n = 20000
    pos_scores = rng.standard_normal(n)
    neg_scores = rng.standard_normal(n)

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    eer, _ = compute_eer(scores, labels)
    assert abs(eer - 0.50) < 0.02, f"Expected EER near 0.50 for random scores, got {eer:.4f}"


def test_analytic_gaussian_eer():
    """Verify empirical EER converges to the exact analytic solution for Gaussians.

    If X_neg ~ N(mu_0, sigma^2) and X_pos ~ N(mu_1, sigma^2),
    by symmetry the optimal equal-error threshold is (mu_0 + mu_1) / 2,
    and analytic EER = 0.5 * (1 - erf((mu_1 - mu_0) / (2 * sqrt(2) * sigma))).
    For mu_0 = -1.0, mu_1 = 1.0, sigma = 1.0:
    Analytic EER = 0.5 * erfc(1.0 / sqrt(2)) = 0.1586552539...
    """
    mu_0, mu_1, sigma = -1.0, 1.0, 1.0
    analytic_eer = 0.5 * math.erfc((mu_1 - mu_0) / (2.0 * math.sqrt(2.0) * sigma))

    rng = np.random.default_rng(42)
    n = 50000
    pos_scores = rng.normal(mu_1, sigma, n)
    neg_scores = rng.normal(mu_0, sigma, n)

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    eer, threshold = compute_eer(scores, labels)

    assert abs(eer - analytic_eer) < 0.005, (
        f"Measured EER {eer:.6f} differs from analytic {analytic_eer:.6f} by more than tolerance"
    )
    # Threshold should be near (mu_0 + mu_1) / 2 = 0.0
    assert abs(threshold - 0.0) < 0.02, f"Threshold {threshold:.4f} not close to theoretical 0.0"


def test_label_direction_inversion():
    """Verify that flipping labels inverts error rates: EER_flipped == 1.0 - EER.

    Speaker verification requires positive (genuine) scores to be systematically
    higher than negative (impostor) scores. If labels are swapped or scores inverted,
    the threshold logic will yield 1.0 - EER.
    """
    rng = np.random.default_rng(999)
    n = 5000
    pos_scores = rng.normal(1.5, 1.0, n)
    neg_scores = rng.normal(-1.5, 1.0, n)

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    eer_normal, _ = compute_eer(scores, labels)

    # Invert labels: 1 becomes 0, 0 becomes 1
    flipped_labels = 1 - labels
    eer_flipped, _ = compute_eer(scores, flipped_labels, check_direction=False)

    assert abs((eer_normal + eer_flipped) - 1.0) < 0.01, (
        f"Flipping labels must give ~100% - EER behavior! Got normal={eer_normal:.4f}, flipped={eer_flipped:.4f}"
    )


@pytest.mark.skipif(not HAS_SPEECHBRAIN, reason="speechbrain not installed")
def test_speechbrain_metric_stats_parity():
    """Verify compute_eer and compute_mindcf match SpeechBrain's reference implementation."""
    rng = np.random.default_rng(2025)
    n = 1000
    pos_scores = rng.normal(1.0, 1.2, n)
    neg_scores = rng.normal(-0.8, 1.2, n)

    pos_t = torch.from_numpy(pos_scores)
    neg_t = torch.from_numpy(neg_scores)

    # Reference SpeechBrain metrics
    sb_eer, sb_eer_th = sb_stats.EER(pos_t, neg_t)
    sb_dcf01, sb_dcf01_th = sb_stats.minDCF(pos_t, neg_t, p_target=0.01)
    sb_dcf05, sb_dcf05_th = sb_stats.minDCF(pos_t, neg_t, p_target=0.05)

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    our_eer, our_eer_th = compute_eer(scores, labels)
    our_dcf01, our_dcf01_th = compute_mindcf(scores, labels, p_target=0.01)
    our_dcf05, our_dcf05_th = compute_mindcf(scores, labels, p_target=0.05)

    # Assert exact or numerical parity
    assert abs(our_eer - sb_eer) < 1e-6, f"EER mismatch: our={our_eer}, sb={sb_eer}"
    assert abs(our_eer_th - sb_eer_th) < 1e-5, f"EER threshold mismatch: our={our_eer_th}, sb={sb_eer_th}"
    assert abs(our_dcf01 - sb_dcf01) < 1e-6, f"minDCF(0.01) mismatch: our={our_dcf01}, sb={sb_dcf01}"
    assert abs(our_dcf05 - sb_dcf05) < 1e-6, f"minDCF(0.05) mismatch: our={our_dcf05}, sb={sb_dcf05}"


def test_det_curve_structure():
    """Verify compute_det_curve returns non-empty arrays with correct monotonic trends."""
    rng = np.random.default_rng(77)
    scores = rng.standard_normal(200)
    labels = rng.integers(0, 2, size=200)

    det = compute_det_curve(scores, labels)
    assert "far" in det and "frr" in det and "thresholds" in det
    far = det["far"]
    frr = det["frr"]
    th = det["thresholds"]

    assert len(far) == len(frr) == len(th)
    # Thresholds are monotonically non-decreasing
    assert np.all(np.diff(th) >= 0)
    # FRR (miss rate) increases with higher threshold
    assert np.all(np.diff(frr) >= 0)
    # FAR (false alarm rate) decreases with higher threshold
    assert np.all(np.diff(far) <= 0)


def test_paired_bootstrap_ci():
    """Verify paired clustered bootstrap computes valid bounds and paired differences."""
    rng = np.random.default_rng(42)
    n_speakers = 20
    trials_per_spk = 25
    total_trials = n_speakers * trials_per_spk

    clusters = np.repeat(np.arange(n_speakers), trials_per_spk)
    labels = rng.integers(0, 2, size=total_trials)

    # System A (better model)
    scores_a = np.where(labels == 1, rng.normal(1.5, 1.0, total_trials), rng.normal(-1.0, 1.0, total_trials))
    # System B (worse model)
    scores_b = np.where(labels == 1, rng.normal(0.5, 1.0, total_trials), rng.normal(-0.5, 1.0, total_trials))

    ci_res = paired_bootstrap_ci(
        scores_a=scores_a,
        labels=labels,
        clusters=clusters,
        scores_b=scores_b,
        n_bootstrap=100,
        confidence_level=0.95,
        seed=42,
    )

    # Check structure
    assert "eer_a_ci" in ci_res
    assert "eer_b_ci" in ci_res
    assert "diff_eer_ci" in ci_res

    low_a, high_a = ci_res["eer_a_ci"]
    low_b, high_b = ci_res["eer_b_ci"]
    diff_low, diff_high = ci_res["diff_eer_ci"]

    assert low_a <= ci_res["eer_a_mean"] <= high_a
    assert low_b <= ci_res["eer_b_mean"] <= high_b
    assert diff_low <= ci_res["diff_eer_mean"] <= diff_high

    # System A should have lower EER than System B, so diff = A - B should be negative
    assert ci_res["diff_eer_mean"] < 0.0


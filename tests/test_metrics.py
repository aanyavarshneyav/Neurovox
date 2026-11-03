"""Unit tests for speaker verification metrics (EER, minDCF, DET, and speaker-level bootstrap).

Verifies numerical correctness, edge cases, input validation, label orientation,
analytic asymptotic limits, and parity against speechbrain.utils.metric_stats across multiple seeds.
"""

import math
import numpy as np
import pytest
import torch

from src.eval.metrics import (
    compute_eer,
    compute_mindcf,
    compute_det_curve,
    speaker_bootstrap_ci,
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

    eer, threshold = compute_eer(scores, labels, strict=True)
    assert eer == 0.0, f"Expected EER 0.0, got {eer}"
    assert -2.0 <= threshold <= 2.0

    res01 = compute_mindcf(scores, labels, p_target=0.01, strict=True)
    res05 = compute_mindcf(scores, labels, p_target=0.05, strict=True)

    assert res01["min_dcf"] == 0.0
    assert res01["min_dcf_raw"] == 0.0
    assert res05["min_dcf"] == 0.0
    assert res05["min_dcf_raw"] == 0.0


def test_random_scores():
    """Verify EER is approximately 50% when positive and negative scores are identical."""
    rng = np.random.default_rng(123)
    n = 20000
    pos_scores = rng.standard_normal(n)
    neg_scores = rng.standard_normal(n)

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    eer, _ = compute_eer(scores, labels, strict=False)
    assert abs(eer - 0.50) < 0.02, f"Expected EER near 0.50 for random scores, got {eer:.4f}"


def test_analytic_gaussian_eer():
    """Verify empirical EER is within 3x sampling SE of the exact analytic Gaussian limit.

    If X_neg ~ N(mu_0, sigma^2) and X_pos ~ N(mu_1, sigma^2),
    analytic EER = 0.5 * erfc((mu_1 - mu_0) / (2 * sqrt(2) * sigma)).
    For mu_0 = -1.0, mu_1 = 1.0, sigma = 1.0:
    Analytic EER = 0.1586552539...
    Sampling SE = sqrt(p * (1 - p) / N).
    """
    mu_0, mu_1, sigma = -1.0, 1.0, 1.0
    analytic_eer = 0.5 * math.erfc((mu_1 - mu_0) / (2.0 * math.sqrt(2.0) * sigma))

    rng = np.random.default_rng(42)
    n = 100000
    pos_scores = rng.normal(mu_1, sigma, n)
    neg_scores = rng.normal(mu_0, sigma, n)

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    eer, threshold = compute_eer(scores, labels, strict=True)

    sampling_se = math.sqrt(analytic_eer * (1.0 - analytic_eer) / n)
    tolerance = 3.0 * sampling_se

    diff = abs(eer - analytic_eer)
    assert diff <= tolerance, (
        f"Measured EER {eer:.6f} differs from analytic {analytic_eer:.6f} by {diff:.6f}, "
        f"exceeding 3x sampling SE ({tolerance:.6f})."
    )
    assert abs(threshold - 0.0) < 0.02, f"Threshold {threshold:.4f} not close to theoretical 0.0"


def test_label_direction_inversion():
    """Verify H1: loud failure on strict=True, and 100% - EER behavior on strict=False."""
    rng = np.random.default_rng(999)
    n = 5000
    pos_scores = rng.normal(1.5, 1.0, n)
    neg_scores = rng.normal(-1.5, 1.0, n)

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    eer_normal, _ = compute_eer(scores, labels, strict=True)

    # Invert labels: targets become lower than impostors
    flipped_labels = 1 - labels

    # Must raise ValueError when strict=True
    with pytest.raises(ValueError, match="Target scores are systematically lower than impostor scores"):
        compute_eer(scores, flipped_labels, strict=True)

    with pytest.raises(ValueError, match="Target scores are systematically lower than impostor scores"):
        compute_mindcf(scores, flipped_labels, strict=True)

    # When strict=False (warning only, e.g. training warmup), returns inverted EER
    eer_flipped, _ = compute_eer(scores, flipped_labels, strict=False)
    assert abs((eer_normal + eer_flipped) - 1.0) < 0.01


@pytest.mark.skipif(not HAS_SPEECHBRAIN, reason="speechbrain not installed")
@pytest.mark.parametrize("seed", [42, 123, 777, 2025, 9999])
def test_speechbrain_metric_stats_parity_across_seeds(seed: int):
    """Verify compute_eer and compute_mindcf match SpeechBrain across >= 5 random seeds."""
    rng = np.random.default_rng(seed)
    n = 800
    pos_scores = rng.normal(1.0, 1.2, n)
    neg_scores = rng.normal(-0.8, 1.2, n)

    pos_t = torch.from_numpy(pos_scores)
    neg_t = torch.from_numpy(neg_scores)

    sb_eer, sb_eer_th = sb_stats.EER(pos_t, neg_t)
    sb_dcf01, sb_dcf01_th = sb_stats.minDCF(pos_t, neg_t, p_target=0.01)
    sb_dcf05, sb_dcf05_th = sb_stats.minDCF(pos_t, neg_t, p_target=0.05)

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    our_eer, our_eer_th = compute_eer(scores, labels, strict=True)
    res01 = compute_mindcf(scores, labels, p_target=0.01, normalize=False, strict=True)
    res05 = compute_mindcf(scores, labels, p_target=0.05, normalize=False, strict=True)

    assert abs(our_eer - sb_eer) < 1e-6, f"Seed {seed} EER mismatch: our={our_eer}, sb={sb_eer}"
    assert abs(our_eer_th - sb_eer_th) < 1e-5, f"Seed {seed} threshold mismatch: our={our_eer_th}, sb={sb_eer_th}"
    assert abs(res01["min_dcf_raw"] - sb_dcf01) < 1e-6, f"Seed {seed} minDCF(0.01) mismatch"
    assert abs(res05["min_dcf_raw"] - sb_dcf05) < 1e-6, f"Seed {seed} minDCF(0.05) mismatch"


def test_input_validation_edge_cases():
    """Verify input validation rejects NaNs, Infs, non-binary labels, and empty classes."""
    valid_scores = np.array([0.5, 0.8, -0.2, -0.4])
    valid_labels = np.array([1, 1, 0, 0])

    # 1. NaN in scores
    nan_scores = valid_scores.copy()
    nan_scores[0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        compute_eer(nan_scores, valid_labels)
    with pytest.raises(ValueError, match="NaN or Inf"):
        compute_mindcf(nan_scores, valid_labels)
    with pytest.raises(ValueError, match="NaN or Inf"):
        compute_det_curve(nan_scores, valid_labels)

    # 2. Inf in scores
    inf_scores = valid_scores.copy()
    inf_scores[1] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        compute_eer(inf_scores, valid_labels)

    # 3. Non-binary labels
    bad_labels = np.array([1, 2, 0, 0])
    with pytest.raises(ValueError, match="binary"):
        compute_eer(valid_scores, bad_labels)

    # 4. Empty class
    all_pos_labels = np.array([1, 1, 1, 1])
    with pytest.raises(ValueError, match="Both positive and negative samples required"):
        compute_eer(valid_scores, all_pos_labels)
    with pytest.raises(ValueError, match="Both positive and negative samples required"):
        compute_det_curve(valid_scores, all_pos_labels)

    # 5. bfloat16 / float16 tensor handling
    if hasattr(torch, "bfloat16"):
        bf16_scores = torch.tensor([0.5, 0.8, -0.2, -0.4], dtype=torch.bfloat16)
        eer, _ = compute_eer(bf16_scores, valid_labels)
        assert 0.0 <= eer <= 1.0


def test_det_curve_structure_and_endpoints():
    """Verify compute_det_curve returns non-empty arrays with accept-all and reject-all endpoints."""
    rng = np.random.default_rng(77)
    scores = rng.standard_normal(200)
    labels = rng.integers(0, 2, size=200)

    det = compute_det_curve(scores, labels)
    far = det["far"]
    frr = det["frr"]
    th = det["thresholds"]

    # Endpoints check:
    # First threshold is accept-all: FAR=1.0, FRR=0.0
    assert far[0] == 1.0 and frr[0] == 0.0, f"Expected accept-all endpoint at start, got FAR={far[0]}, FRR={frr[0]}"
    # Last threshold is reject-all: FAR=0.0, FRR=1.0
    assert far[-1] == 0.0 and frr[-1] == 1.0, f"Expected reject-all endpoint at end, got FAR={far[-1]}, FRR={frr[-1]}"

    # Monotonicity
    assert np.all(np.diff(th) >= 0)
    assert np.all(np.diff(frr) >= 0)
    assert np.all(np.diff(far) <= 0)


def test_speaker_bootstrap_ci():
    """Verify speaker-level clustered bootstrap computes point estimates, CIs, and paired diffs."""
    rng = np.random.default_rng(42)
    n_speakers = 10
    spk_ids = [f"spk_{i:02d}" for i in range(n_speakers)]

    # Form trials between speakers
    trials_records = []
    # Target trials: 20 per speaker
    for s in spk_ids:
        for _ in range(20):
            trials_records.append((1, s, s, rng.normal(1.2, 0.8), rng.normal(0.4, 0.8)))
    # Impostor trials: 20 per speaker pair
    for i in range(n_speakers):
        for j in range(i + 1, n_speakers):
            for _ in range(4):
                s1, s2 = spk_ids[i], spk_ids[j]
                trials_records.append((0, s1, s2, rng.normal(-0.8, 0.8), rng.normal(-0.3, 0.8)))

    labels = np.array([r[0] for r in trials_records], dtype=int)
    spk_a = np.array([r[1] for r in trials_records])
    spk_b = np.array([r[2] for r in trials_records])
    scores_a = np.array([r[3] for r in trials_records])
    scores_b = np.array([r[4] for r in trials_records])

    ci_res = speaker_bootstrap_ci(
        scores_a=scores_a,
        labels=labels,
        spk_a=spk_a,
        spk_b=spk_b,
        scores_b=scores_b,
        n_bootstrap=100,
        confidence_level=0.95,
        seed=42,
    )

    assert "point_estimate_a" in ci_res
    assert "bootstrap_mean_a" in ci_res
    assert "ci_95_a" in ci_res
    assert "point_estimate_b" in ci_res
    assert "diff_ci_95" in ci_res

    pt_a = ci_res["point_estimate_a"]["eer"]
    low_a, high_a = ci_res["ci_95_a"]["eer"]
    assert low_a <= pt_a <= high_a or abs(pt_a - low_a) < 0.05

    # Difference CI: System A is better than System B, diff = A - B should be negative
    diff_pt = ci_res["diff_point_estimate"]["diff_eer"]
    diff_low, diff_high = ci_res["diff_ci_95"]["diff_eer"]
    assert diff_pt < 0.0
    assert diff_low <= diff_high


def test_speaker_bootstrap_two_sample_difference():
    """Verify two-sample speaker bootstrap across independent sets computes point estimates and difference CIs."""
    from src.eval.metrics import speaker_bootstrap_two_sample_difference

    rng = np.random.default_rng(123)

    def _make_set(n_spks, spk_prefix, pos_mean, neg_mean):
        spks = [f"{spk_prefix}_{i}" for i in range(n_spks)]
        records = []
        for s in spks:
            for _ in range(15):
                records.append((1, s, s, rng.normal(pos_mean, 0.6)))
        for i in range(n_spks):
            for j in range(i + 1, n_spks):
                records.append((0, spks[i], spks[j], rng.normal(neg_mean, 0.6)))
        lbls = np.array([r[0] for r in records], dtype=int)
        sa = np.array([r[1] for r in records])
        sb = np.array([r[2] for r in records])
        sc = np.array([r[3] for r in records])
        return sc, lbls, sa, sb

    # Set 1 (e.g. test-clean, higher separation)
    sc1, lbl1, sa1, sb1 = _make_set(8, "clean", 1.5, -1.0)
    # Set 2 (e.g. test-other, lower separation)
    sc2, lbl2, sa2, sb2 = _make_set(8, "other", 0.8, -0.5)

    res = speaker_bootstrap_two_sample_difference(
        scores_1=sc1,
        labels_1=lbl1,
        spk_a_1=sa1,
        spk_b_1=sb1,
        scores_2=sc2,
        labels_2=lbl2,
        spk_a_2=sa2,
        spk_b_2=sb2,
        n_bootstrap=100,
        confidence_level=0.95,
        seed=42,
    )

    assert "point_estimate_1" in res
    assert "point_estimate_2" in res
    assert "diff_point_estimate" in res
    assert "diff_ci_95" in res

    diff_eer = res["diff_point_estimate"]["diff_eer"]
    # Set 2 has higher EER than Set 1, so Set 2 - Set 1 > 0
    assert diff_eer > 0.0
    low, high = res["diff_ci_95"]["diff_eer"]
    assert low <= high

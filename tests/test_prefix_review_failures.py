"""Pre-fix test suite for review findings H1, H2, H4, and H5.

These tests assert the required new behaviors. They MUST FAIL on the old code,
proving that the issues were real and that the tests are sensitive to them.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.utils.paths import get_data_root
from src.eval.metrics import compute_eer, compute_mindcf
from src.data.trial_generator import TrialGenerator


def test_h1_direction_check_fails_loudly_on_inverted_scores():
    """H1: When mean(target) < mean(impostor), compute_eer must raise ValueError when strict=True."""
    pos_scores = np.array([-1.0, -0.5, -0.8])
    neg_scores = np.array([1.0, 0.8, 1.2])

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])

    # Must raise ValueError when strict=True
    with pytest.raises(ValueError, match="Target scores are systematically lower than impostor scores"):
        compute_eer(scores, labels, strict=True)


def test_h2_mindcf_is_normalized_by_default_with_named_keys():
    """H2: minDCF must be normalized by default and return both normalized and raw under named keys."""
    # With overlapping scores, minDCF is strictly positive
    pos_scores = np.array([0.1, 0.4, 0.8, 1.2])
    neg_scores = np.array([-0.2, 0.2, 0.5, 0.7])
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])

    p_target = 0.01
    c_miss = 1.0
    c_fa = 1.0
    c_def = min(c_miss * p_target, c_fa * (1.0 - p_target))  # 0.01

    res = compute_mindcf(scores, labels, p_target=p_target, c_miss=c_miss, c_fa=c_fa)

    # Must support named keys
    assert isinstance(res, dict) or hasattr(res, "min_dcf"), "Must return result with named keys"
    assert "min_dcf" in res, "Missing 'min_dcf' (normalized) key"
    assert "min_dcf_raw" in res, "Missing 'min_dcf_raw' (unnormalized) key"
    assert "threshold" in res, "Missing 'threshold' key"

    # Default min_dcf must be NORMALIZED
    assert res["min_dcf"] == pytest.approx(res["min_dcf_raw"] / c_def)
    assert res["min_dcf"] > res["min_dcf_raw"], "Normalized minDCF must exceed raw minDCF for small P_target"


def test_h4_impostor_gender_counts_must_match_target_gender_counts():
    """H4: Generator must match impostor gender counts to target gender counts per set."""
    manifest_path = get_data_root() / "manifests" / "test-clean.csv"
    if not manifest_path.exists():
        pytest.skip("Manifest test-clean.csv not found.")

    gen = TrialGenerator(manifest_path=manifest_path, min_duration=2.0, seed=42)
    trials, report = gen.generate_trials()

    # Read trial paths and speaker genders from manifest
    df_manifest = pd.read_csv(manifest_path)
    spk_to_gender = dict(zip(df_manifest["spk_id"].astype(str), df_manifest["gender"]))

    target_genders = {"M": 0, "F": 0}
    impostor_genders = {"M": 0, "F": 0}

    for label, p1, p2 in trials:
        spk1 = Path(p1).parts[-3]
        g1 = spk_to_gender[spk1]
        if label == 1:
            target_genders[g1] += 1
        else:
            impostor_genders[g1] += 1

    # In test-clean, targets are 1300 M / 1600 F. Impostors must match this distribution!
    assert impostor_genders["M"] == target_genders["M"], (
        f"Impostor male count {impostor_genders['M']} does not match target male count {target_genders['M']}"
    )
    assert impostor_genders["F"] == target_genders["F"], (
        f"Impostor female count {impostor_genders['F']} does not match target female count {target_genders['F']}"
    )


def test_h5_mean_durations_target_vs_impostor_differ_by_less_than_3_percent():
    """H5: Mean duration of target and impostor trials must differ by < 3%."""
    manifest_path = get_data_root() / "manifests" / "test-clean.csv"
    if not manifest_path.exists():
        pytest.skip("Manifest test-clean.csv not found.")

    gen = TrialGenerator(manifest_path=manifest_path, min_duration=2.0, seed=42)
    trials, report = gen.generate_trials()

    df_manifest = pd.read_csv(manifest_path)
    path_to_dur = dict(zip(df_manifest["path"], df_manifest["duration"]))

    target_durs = []
    impostor_durs = []

    for label, p1, p2 in trials:
        avg_dur = (path_to_dur[p1] + path_to_dur[p2]) / 2.0
        if label == 1:
            target_durs.append(avg_dur)
        else:
            impostor_durs.append(avg_dur)

    mean_target = np.mean(target_durs)
    mean_impostor = np.mean(impostor_durs)

    rel_diff = abs(mean_target - mean_impostor) / mean_target
    assert rel_diff < 0.03, (
        f"Target mean dur ({mean_target:.2f}s) vs impostor ({mean_impostor:.2f}s) "
        f"differs by {rel_diff*100:.2f}% (must be < 3%)"
    )

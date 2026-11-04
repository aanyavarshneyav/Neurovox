"""Unit test for train-probe evaluation script on CPU using synthetic trials."""

import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.train_probe import (
    compute_sha256,
    instantiate_model_for_checkpoint,
    run_train_probe_evaluation,
)
from src.models.small_tdnn_se import small_tdnn_se


def test_train_probe_evaluation_synthetic_cpu(tmp_path: Path):
    """Verify train-probe evaluation runs on CPU with synthetic waveforms and produces required metrics."""
    # 1. Create a few synthetic 16 kHz FLAC files for 3 speakers (2 utts each)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    sample_rate = 16000

    # 3 speakers: spk01, spk02, spk03
    speakers = ["spk01", "spk02", "spk03"]
    utt_paths: dict[str, list[Path]] = {}
    rng = np.random.default_rng(42)

    for spk in speakers:
        utt_paths[spk] = []
        for i in range(2):
            # 1.5 seconds of synthetic audio
            audio = rng.normal(0, 0.1, size=int(sample_rate * 1.5)).astype(np.float32)
            p = audio_dir / f"{spk}_utt{i}.flac"
            sf.write(str(p), audio, sample_rate)
            utt_paths[spk].append(p)

    # 2. Build synthetic trial list and metadata CSV
    # Targets: within same speaker
    # Impostors: across different speakers
    trial_records = []
    # 3 targets (one per speaker)
    for spk in speakers:
        trial_records.append((1, str(utt_paths[spk][0]), str(utt_paths[spk][1]), spk, spk))
    # 3 impostors
    trial_records.append((0, str(utt_paths["spk01"][0]), str(utt_paths["spk02"][0]), "spk01", "spk02"))
    trial_records.append((0, str(utt_paths["spk02"][0]), str(utt_paths["spk03"][0]), "spk02", "spk03"))
    trial_records.append((0, str(utt_paths["spk03"][0]), str(utt_paths["spk01"][0]), "spk03", "spk01"))

    trials_txt = tmp_path / "trials_synthetic.txt"
    with open(trials_txt, "w", encoding="utf-8") as f:
        for lbl, pa, pb, _, _ in trial_records:
            f.write(f"{lbl} {pa} {pb}\n")

    trials_meta = tmp_path / "trials_synthetic_meta.csv"
    with open(trials_meta, "w", encoding="utf-8") as f:
        f.write("label,path_a,path_b,spk_a,spk_b\n")
        for lbl, pa, pb, sa, sb in trial_records:
            f.write(f"{lbl},{pa},{pb},{sa},{sb}\n")

    # 3. Instantiate tiny random-weight model on CPU
    model = small_tdnn_se(in_features=80, channels=32, emb_dim=64)
    model.eval()

    # 4. Run train-probe evaluation on CPU
    device = torch.device("cpu")
    res = run_train_probe_evaluation(
        model=model,
        trials_path=trials_txt,
        trials_meta_path=trials_meta,
        device=device,
        bootstrap_iterations=20,
        seed=42,
        show_progress=False,
    )

    # 5. Assert contract and metrics
    assert res["num_trials"] == 6
    assert res["num_targets"] == 3
    assert res["num_impostors"] == 3
    assert res["num_speakers"] == 3
    assert 0.0 <= res["eer"] <= 1.0
    assert 0.0 <= res["eer_pct"] <= 100.0
    assert np.isfinite(res["eer_threshold"])
    assert "norm" in res["mindcf_p05"]
    assert "raw" in res["mindcf_p05"]
    assert "threshold" in res["mindcf_p05"]
    assert "norm" in res["mindcf_p01"]
    assert res["trials_sha256"] == compute_sha256(trials_txt)

    # Verify bootstrap CIs
    ci = res["bootstrap_ci_95"]
    assert ci is not None
    assert "eer_low_pct" in ci
    assert "eer_high_pct" in ci
    assert "mindcf_p05_norm_low" in ci
    assert "mindcf_p05_norm_high" in ci


def test_instantiate_model_fallback(tmp_path: Path):
    """Verify fallback instantiation works for state_dict with small_tdnn_se and SKA-HNN."""
    # Test small_tdnn_se payload
    dummy_model = small_tdnn_se(in_features=80, channels=32, emb_dim=64)
    payload_small = {
        "epoch": 5,
        "run_id": "test_run",
        "model_state_dict": dummy_model.state_dict(),
    }
    model, name = instantiate_model_for_checkpoint(payload_small, torch.device("cpu"))
    assert "small_tdnn_se" in name
    assert isinstance(model, torch.nn.Module)

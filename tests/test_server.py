"""Integration test for the Stage 1 Neurovox FastAPI verification server."""

import io
import json
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf
import torch

from app.server import (
    STATE,
    _derive_eval_interval,
    app,
    classify_verdict,
    compute_audio_rms,
    compute_file_sha256,
    compute_or_load_threshold,
)


def _make_wav_bytes(duration_sec: float = 1.5, sr: int = 16000, seed: int = 0) -> bytes:
    """Create in-memory 16 kHz mono WAV bytes of the requested duration."""
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(int(sr * duration_sec)).astype(np.float32) * 0.1
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_server_loads_checkpoint_and_verifies_two_short_clips():
    """Load the checkpoint through the server code and check /api/verify returns score in [-1, 1]."""
    STATE.initialize(device_str="cpu")

    with TestClient(app) as client:
        info_resp = client.get("/api/info")
        assert info_resp.status_code == 200
        info = info_resp.json()
        assert info["model_name"]
        assert info["parameter_count"] > 0
        assert info["checkpoint_path"]
        assert info["checkpoint_epoch"] >= 1
        assert abs(float(info["threshold"]) - 0.5944) <= 0.0005
        assert abs(float(info["borderline_low"]) - 0.4944) <= 0.0005
        assert abs(float(info["borderline_high"]) - 0.6944) <= 0.0005
        assert "Scaled reproduction" in info["model_label"]
        assert info["checkpoint_sha256"]
        assert info["borderline_zone"] == "threshold +/- 0.10"
        assert float(info["borderline_width"]) == 0.20

        wav_enroll = _make_wav_bytes(duration_sec=1.5, seed=42)
        wav_test = _make_wav_bytes(duration_sec=1.5, seed=99)

        verify_resp = client.post(
            "/api/verify",
            files={
                "enrollment": ("enroll.wav", wav_enroll, "audio/wav"),
                "test": ("test.wav", wav_test, "audio/wav"),
            },
        )
        assert verify_resp.status_code == 200, verify_resp.text
        data = verify_resp.json()

        score = float(data["score"])
        assert -1.0 <= score <= 1.0
        assert data["verdict"] in {"SAME", "DIFFERENT", "INCONCLUSIVE"}
        assert data["borderline_zone"] == "threshold +/- 0.10"
        assert float(data["borderline_width"]) == 0.20
        assert float(data["enrollment_duration"]) >= 1.0
        assert float(data["test_duration"]) >= 1.0

        # Check verdict band helper directly
        assert classify_verdict(0.75, 0.50, 0.10) == "SAME"
        assert classify_verdict(0.55, 0.50, 0.10) == "INCONCLUSIVE"
        assert classify_verdict(0.45, 0.50, 0.10) == "INCONCLUSIVE"
        assert classify_verdict(0.35, 0.50, 0.10) == "DIFFERENT"

        # Check dynamic eval_interval derivation
        assert _derive_eval_interval({"eval_interval": 3}, [{"epoch": 2, "dev_eer_pct": 15.0}]) == 3
        assert _derive_eval_interval(
            {},
            [
                {"epoch": 1, "dev_eer_pct": None},
                {"epoch": 2, "dev_eer_pct": 16.71},
                {"epoch": 3, "dev_eer_pct": None},
                {"epoch": 4, "dev_eer_pct": 15.23},
            ],
        ) == 2

        # Clips shorter than 1.0s must be rejected with a clear JSON error
        short_wav = _make_wav_bytes(duration_sec=0.4, seed=7)
        short_resp = client.post(
            "/api/verify",
            files={
                "enrollment": ("short.wav", short_wav, "audio/wav"),
                "test": ("test.wav", wav_test, "audio/wav"),
            },
        )
        assert short_resp.status_code == 400
        assert "too short" in short_resp.json().get("error", "").lower()

        # Check Stage 2 /api/results endpoint
        results_resp = client.get("/api/results")
        assert results_resp.status_code == 200
        res_data = results_resp.json()
        assert res_data["runs"]["s0"] is not None
        assert len(res_data["runs"]["s0"]["epochs"]) >= 1
        assert res_data["runs"]["s0"]["eval_interval"] == 2
        assert len(res_data["architecture"]["clickable_modules"]) >= 9
        assert len(res_data["architecture"]["ablation_switches"]) == 7
        assert len(res_data["tiers"]) == 5
        colab_wc = res_data["benchmarks"]["colab_pilot_wallclock"]
        assert [c["total_sec"] for c in colab_wc] == [637.21, 638.17, 651.02]
        assert all(
            c["platform_label"]
            == "Colab T4, whole-run wall-clock (old code, before timing instrumentation; no breakdown)"
            and c["label_type"] == "measured"
            for c in colab_wc
        )
        assert all("colab_peak_vram_mb" not in r for r in res_data["benchmarks"]["hardware_throughput"])
        assert "found" in res_data["sources"]
        assert "missing" in res_data["sources"]


def test_threshold_cache_stale_sha_rejected(tmp_path: Path, monkeypatch):
    """Verify that threshold cache is rejected when SHA-256 does not match checkpoint."""
    dummy_ckpt = tmp_path / "model.pt"
    dummy_ckpt.write_bytes(b"dummy checkpoint content for sha test")
    real_sha = compute_file_sha256(dummy_ckpt)

    cache_path = tmp_path / "model.threshold.json"

    # 1. Stale cache with mismatched sha256
    cache_path.write_text(
        json.dumps({
            "sha256": "stale_hash_that_does_not_match",
            "threshold": 0.1111,
            "epoch": 5,
        }),
        encoding="utf-8",
    )

    # Mock evaluate_trials return to see if recomputation happens
    mock_model = torch.nn.Linear(10, 10)
    called_eval = []

    def fake_evaluate_trials(**kwargs):
        called_eval.append(True)
        return {"eer_threshold": 0.5944, "eer_pct": 12.06}

    monkeypatch.setattr("app.server.evaluate_trials", fake_evaluate_trials)

    thresh, dev_eer, epoch, path, sha = compute_or_load_threshold(
        model=mock_model,
        ckpt_path=dummy_ckpt,
        ckpt_payload={"epoch": 30, "best_dev_eer": 12.06},
        device=torch.device("cpu"),
        trials_path=dummy_ckpt,  # dummy path
    )
    # Should have called evaluate_trials because stale sha was rejected
    assert len(called_eval) == 1
    assert thresh == 0.5944
    assert sha == real_sha

    # The cache file should now be rewritten with the new real_sha
    with open(cache_path, "r", encoding="utf-8") as f:
        updated_cache = json.load(f)
    assert updated_cache["sha256"] == real_sha
    assert updated_cache["threshold"] == 0.5944

    # 2. Second load with valid matching sha must NOT call evaluate_trials again
    called_eval.clear()
    thresh2, _, _, _, sha2 = compute_or_load_threshold(
        model=mock_model,
        ckpt_path=dummy_ckpt,
        ckpt_payload={"epoch": 30, "best_dev_eer": 12.06},
        device=torch.device("cpu"),
        trials_path=dummy_ckpt,
    )
    assert len(called_eval) == 0  # Cache hit!
    assert thresh2 == 0.5944
    assert sha2 == real_sha


def test_silence_guard_rejects_zero_and_low_energy():
    """Verify that /api/verify rejects zero-filled audio and very quiet noise with HTTP 400."""
    STATE.initialize(device_str="cpu")

    with TestClient(app) as client:
        normal_wav = _make_wav_bytes(duration_sec=1.5, seed=42)

        # 1. Zero-filled enrollment
        zero_samples = np.zeros(16000 * 2, dtype=np.float32)
        zero_buf = io.BytesIO()
        sf.write(zero_buf, zero_samples, 16000, format="WAV", subtype="PCM_16")
        zero_wav = zero_buf.getvalue()

        r1 = client.post(
            "/api/verify",
            files={
                "enrollment": ("zero.wav", zero_wav, "audio/wav"),
                "test": ("norm.wav", normal_wav, "audio/wav"),
            },
        )
        assert r1.status_code == 400
        assert r1.json()["error"] == "No speech detected in enrollment clip."

        # 2. Very quiet noise enrollment (std=1e-4, RMS ~ 0.0001 < 0.005)
        rng = np.random.default_rng(123)
        quiet_samples = rng.standard_normal(16000 * 2).astype(np.float32) * 1e-4
        quiet_buf = io.BytesIO()
        sf.write(quiet_buf, quiet_samples, 16000, format="WAV", subtype="PCM_16")
        quiet_wav = quiet_buf.getvalue()

        r2 = client.post(
            "/api/verify",
            files={
                "enrollment": ("quiet.wav", quiet_wav, "audio/wav"),
                "test": ("norm.wav", normal_wav, "audio/wav"),
            },
        )
        assert r2.status_code == 400
        assert r2.json()["error"] == "No speech detected in enrollment clip."

        # 3. Zero-filled test clip
        r3 = client.post(
            "/api/verify",
            files={
                "enrollment": ("norm.wav", normal_wav, "audio/wav"),
                "test": ("zero.wav", zero_wav, "audio/wav"),
            },
        )
        assert r3.status_code == 400
        assert r3.json()["error"] == "No speech detected in test clip."

        # 4. Normal clip passes speech detection
        r4 = client.post(
            "/api/verify",
            files={
                "enrollment": ("norm1.wav", normal_wav, "audio/wav"),
                "test": ("norm2.wav", _make_wav_bytes(duration_sec=1.5, seed=99), "audio/wav"),
            },
        )
        assert r4.status_code == 200


def test_length_cap_rejects_over_30s():
    """Verify that clips longer than 30s are rejected with HTTP 400."""
    STATE.initialize(device_str="cpu")

    with TestClient(app) as client:
        normal_wav = _make_wav_bytes(duration_sec=1.5, seed=42)
        long_wav = _make_wav_bytes(duration_sec=31.0, seed=101)

        # 1. Enrollment clip too long
        r1 = client.post(
            "/api/verify",
            files={
                "enrollment": ("long.wav", long_wav, "audio/wav"),
                "test": ("norm.wav", normal_wav, "audio/wav"),
            },
        )
        assert r1.status_code == 400
        assert "too long" in r1.json()["error"].lower()
        assert "30.0s" in r1.json()["error"]

        # 2. Test clip too long
        r2 = client.post(
            "/api/verify",
            files={
                "enrollment": ("norm.wav", normal_wav, "audio/wav"),
                "test": ("long.wav", long_wav, "audio/wav"),
            },
        )
        assert r2.status_code == 400
        assert "too long" in r2.json()["error"].lower()
        assert "30.0s" in r2.json()["error"]



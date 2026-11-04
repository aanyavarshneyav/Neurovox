"""Integration test for the Stage 1 Neurovox FastAPI verification server."""

import io
from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf

from app.server import STATE, app, classify_verdict


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
        assert -1.0 <= float(info["threshold"]) <= 1.0
        assert info["borderline_zone"] == "threshold +/- 0.10"

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
        assert float(data["enrollment_duration"]) >= 1.0
        assert float(data["test_duration"]) >= 1.0

        # Check verdict band helper directly
        assert classify_verdict(0.75, 0.50, 0.10) == "SAME"
        assert classify_verdict(0.55, 0.50, 0.10) == "INCONCLUSIVE"
        assert classify_verdict(0.45, 0.50, 0.10) == "INCONCLUSIVE"
        assert classify_verdict(0.35, 0.50, 0.10) == "DIFFERENT"

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
        assert len(res_data["architecture"]["clickable_modules"]) >= 9
        assert len(res_data["architecture"]["ablation_switches"]) == 7
        assert len(res_data["tiers"]) == 5
        assert "found" in res_data["sources"]
        assert "missing" in res_data["sources"]


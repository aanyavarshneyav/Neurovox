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
        assert info["checkpoint_path"] == "checkpoints/s0_seed44/best_dev_eer.pt"
        assert info["checkpoint_epoch"] == 28
        assert info["seed"] == 44
        assert abs(float(info["dev_eer"]) - 11.61) <= 0.01
        assert abs(float(info["threshold"]) - 0.5755) <= 0.0005
        assert abs(float(info["borderline_low"]) - 0.4755) <= 0.0005
        assert abs(float(info["borderline_high"]) - 0.6755) <= 0.0005
        assert info["checkpoint_sha256"] == "4e445c042951d7c83b4f7ce48c1f5c0feb1e618f18d4b45ee495d87a094460db"
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


def _write_mock_run_csv(
    run_dir: Path,
    epochs: int,
    best_eer: float,
    best_mindcf: float,
    seed: int = 42,
) -> None:
    """Write a minimal valid history.csv and run_meta.json into a temporary fixture directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_dir.name,
        "seed": seed,
        "device": "mps",
        "precision": "fp32",
        "parameter_count": 4167700,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    lines = ["epoch,train_loss,train_acc,dev_eer_pct,dev_mindcf_05,total_sec"]
    for ep in range(1, epochs + 1):
        if ep % 2 == 0:
            eer = best_eer if ep == epochs else best_eer + 1.5
            dcf = best_mindcf if ep == epochs else best_mindcf + 0.05
            lines.append(f"{ep},2.5000,60.00,{eer:.2f},{dcf:.4f},240.0")
        else:
            lines.append(f"{ep},2.6000,58.00,,,220.0")
    (run_dir / "history.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_dashboard_measured_and_projected_separation():
    """Verify /api/results separates measured and projected data and follows schema rules."""
    STATE.initialize(device_str="cpu")
    with TestClient(app) as client:
        res = client.get("/api/results")
        assert res.status_code == 200
        payload = res.json()

        assert "measured" in payload
        assert "projected" in payload
        meas = payload["measured"]
        proj = payload["projected"]

        assert meas["status"] == "measured"
        assert proj["status"] == "projected"

        # Section A: Headline S0 vs S1 contains only measured data
        headline = meas["headline_s0_s1"]
        assert 42 in headline["completed_seeds"]
        assert 43 in headline["completed_seeds"]
        assert 44 in headline["completed_seeds"]
        assert headline["summary"]["s0"]["status"] == "measured"
        assert headline["summary"]["s1"]["status"] == "measured"
        assert headline["summary"]["s0"]["parameter_count"] == 4167700

        # Section D: S4 must be excluded; S0, S1, S2, S3, S5, S6 present with measured parameter counts
        ablation_switches = [a["switch"] for a in meas["ablations"]]
        assert ablation_switches == ["S0", "S1", "S2", "S3", "S5", "S6"]
        assert "S4" not in ablation_switches
        for ab in meas["ablations"]:
            assert ab["parameter_status"] == "measured"
            assert isinstance(ab["parameter_count"], int) and ab["parameter_count"] > 0

        # Projected entries record value, unit, metric, status, formula, inputs, caveat
        for sw in ("S2", "S3", "S5", "S6"):
            entry = proj["ablations"][sw]["eer_pct"]
            assert entry["status"] == "projected"
            assert entry["value"] is not None
            assert "formula" in entry and "*" in entry["formula"]
            assert "inputs" in entry and "our_anchor" in entry["inputs"]
            assert "caveat" in entry

        # Section B: 10%, 20%, 40% synthetic noise projections are populated with status="projected"
        for lvl in ("10", "20", "40"):
            assert proj["label_noise"][lvl]["s0"]["eer_pct"]["status"] == "projected"
            assert proj["label_noise"][lvl]["s0"]["eer_pct"]["value"] is not None
            assert proj["label_noise"][lvl]["s1"]["eer_pct"]["status"] == "projected"
            assert proj["label_noise"][lvl]["s1"]["eer_pct"]["value"] is not None


def test_seed44_provisional_vs_completed_integration(tmp_path: Path):
    """Verify incomplete seed 44 is excluded from mean/SD and automatically included once completed."""
    from app.server import _build_dashboard_payload, _collect_results_runs

    results_root = tmp_path / "results"
    placeholders_dir = results_root / "placeholders"
    placeholders_dir.mkdir(parents=True, exist_ok=True)

    # Copy paper_reference.json into temporary fixture
    real_paper_ref = Path(__file__).resolve().parent.parent / "results" / "placeholders" / "paper_reference.json"
    (placeholders_dir / "paper_reference.json").write_text(
        real_paper_ref.read_text(encoding="utf-8"), encoding="utf-8"
    )

    # Completed seeds 42 and 43, plus incomplete seed 44 (8/30 epochs for S0, S1 not started)
    _write_mock_run_csv(results_root / "s0_seed42", epochs=30, best_eer=12.06, best_mindcf=0.4594, seed=42)
    _write_mock_run_csv(results_root / "s1_seed42", epochs=30, best_eer=11.71, best_mindcf=0.4506, seed=42)
    _write_mock_run_csv(results_root / "s0_seed43", epochs=30, best_eer=11.77, best_mindcf=0.4390, seed=43)
    _write_mock_run_csv(results_root / "s1_seed43", epochs=30, best_eer=11.71, best_mindcf=0.4729, seed=43)
    _write_mock_run_csv(results_root / "s0_seed44", epochs=8, best_eer=13.50, best_mindcf=0.5200, seed=44)

    all_runs, found, missing = _collect_results_runs(results_roots=[results_root])
    meas1, proj1 = _build_dashboard_payload(all_runs, found, missing, placeholders_dir=placeholders_dir)

    h1 = meas1["headline_s0_s1"]
    assert h1["completed_seeds"] == [42, 43]
    assert h1["pending_seeds"] == [44]
    assert abs(h1["summary"]["s0"]["eer_mean_pct"] - 11.915) < 1e-4
    seed44_rec = next(s for s in h1["seeds"] if s["seed"] == 44)
    assert seed44_rec["pair_completed"] is False
    assert seed44_rec["is_provisional"] is True

    # Now complete both s0_seed44 and s1_seed44 (30 epochs each)
    _write_mock_run_csv(results_root / "s0_seed44", epochs=30, best_eer=11.87, best_mindcf=0.4492, seed=44)
    _write_mock_run_csv(results_root / "s1_seed44", epochs=30, best_eer=11.68, best_mindcf=0.4600, seed=44)

    all_runs2, found2, missing2 = _collect_results_runs(results_roots=[results_root])
    meas2, proj2 = _build_dashboard_payload(all_runs2, found2, missing2, placeholders_dir=placeholders_dir)

    h2 = meas2["headline_s0_s1"]
    assert h2["completed_seeds"] == [42, 43, 44]
    assert h2["pending_seeds"] == []
    expected_s0_mean = round((12.06 + 11.77 + 11.87) / 3.0, 4)
    assert abs(h2["summary"]["s0"]["eer_mean_pct"] - expected_s0_mean) < 1e-4
    seed44_rec2 = next(s for s in h2["seeds"] if s["seed"] == 44)
    assert seed44_rec2["pair_completed"] is True
    assert seed44_rec2["is_provisional"] is False
    assert proj2["measured_anchors_snapshot"]["completed_seeds"] == [42, 43, 44]


def test_missing_data_and_auto_replacement_in_tmp_path(tmp_path: Path):
    """Verify empty-directory safety and automatic suppression of projections when measured data exists."""
    from app.server import _build_dashboard_payload, _collect_results_runs

    empty_root = tmp_path / "empty_results"
    empty_root.mkdir(parents=True, exist_ok=True)
    all_runs_empty, found_e, missing_e = _collect_results_runs(results_roots=[empty_root])
    meas_e, proj_e = _build_dashboard_payload(
        all_runs_empty, found_e, missing_e, placeholders_dir=empty_root / "placeholders"
    )
    assert meas_e["headline_s0_s1"]["completed_seeds"] == []
    assert meas_e["headline_s0_s1"]["summary"]["s0"]["eer_mean_pct"] is None
    assert proj_e["available"] is False

    # Now populate mock measured runs for S0/S1 seed 42, plus S2 ablation, ECAPA, and 10% noise
    results_root = tmp_path / "populated_results"
    placeholders_dir = results_root / "placeholders"
    placeholders_dir.mkdir(parents=True, exist_ok=True)
    real_paper_ref = Path(__file__).resolve().parent.parent / "results" / "placeholders" / "paper_reference.json"
    (placeholders_dir / "paper_reference.json").write_text(
        real_paper_ref.read_text(encoding="utf-8"), encoding="utf-8"
    )

    _write_mock_run_csv(results_root / "s0_seed42", epochs=30, best_eer=12.00, best_mindcf=0.4500, seed=42)
    _write_mock_run_csv(results_root / "s1_seed42", epochs=30, best_eer=11.80, best_mindcf=0.4600, seed=42)
    _write_mock_run_csv(results_root / "s2_seed42", epochs=30, best_eer=12.40, best_mindcf=0.4800, seed=42)
    _write_mock_run_csv(results_root / "ecapa_seed42", epochs=30, best_eer=13.10, best_mindcf=0.5100, seed=42)
    _write_mock_run_csv(results_root / "s0_noise10_seed42", epochs=30, best_eer=14.20, best_mindcf=0.5300, seed=42)

    all_runs_pop, found_p, missing_p = _collect_results_runs(results_roots=[results_root])
    meas_p, proj_p = _build_dashboard_payload(
        all_runs_pop, found_p, missing_p, placeholders_dir=placeholders_dir
    )

    # S2 is now measured and its projection is suppressed
    s2_meas = next(a for a in meas_p["ablations"] if a["switch"] == "S2")
    assert s2_meas["status"] == "measured"
    assert s2_meas["eer_pct"] == 12.40
    assert proj_p["ablations"]["S2"]["suppressed_by_measured"] is True

    # S3 is still not_run and its projection is active (not suppressed)
    s3_meas = next(a for a in meas_p["ablations"] if a["switch"] == "S3")
    assert s3_meas["status"] == "not_run"
    assert proj_p["ablations"]["S3"]["suppressed_by_measured"] is False

    # ECAPA is now measured and its projection is suppressed
    assert meas_p["model_comparison"]["ecapa_tdnn"]["status"] == "measured"
    assert meas_p["model_comparison"]["ecapa_tdnn"]["eer_mean_pct"] == 13.10
    assert proj_p["ecapa_comparison"]["suppressed_by_measured"] is True

    # 10% label noise for S0 is now measured and its projection is suppressed
    assert meas_p["label_noise"]["10"]["s0"]["status"] == "measured"
    assert meas_p["label_noise"]["10"]["s0"]["eer_pct"] == 14.20
    assert proj_p["label_noise"]["10"]["s0"]["suppressed_by_measured"] is True




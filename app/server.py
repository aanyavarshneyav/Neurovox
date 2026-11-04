#!/usr/bin/env python3
"""FastAPI backend for Stage 1 of the Neurovox speaker verification demo site.

Features:
- Loads trained SKA-HNN (or baseline) checkpoint on CPU by default (--device to override).
- Reuses existing model factory, 80-dim log-Mel frontend, and full-utterance EmbeddingExtractor.
- Computes dev-clean EER threshold once at startup via scripts.evaluate.evaluate_trials
  and caches it to a JSON file next to the checkpoint (recomputed if missing).
- Decodes browser audio blobs (webm/ogg/wav/flac) to 16 kHz mono float32 entirely in memory
  via ffmpeg pipes (never writing audio to disk).
- Exposes GET /api/info, POST /api/verify, GET /api/examples, and serves app/static/index.html.
"""

import argparse
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple, Union

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch
import torchinfo
import uvicorn

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure standard macOS Homebrew binary locations are on PATH so shutil.which("ffmpeg") succeeds
for _extra_bin in ("/opt/homebrew/bin", "/usr/local/bin"):
    if _extra_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{os.environ.get('PATH', '')}:{_extra_bin}"

from scripts.evaluate import evaluate_trials
from scripts.train import compute_config_hash, load_config
from src.eval.extractor import EmbeddingExtractor
from src.models import (
    ECAPATDNN,
    build_ska_hnn,
    build_ska_hnn_from_config,
    ecapa_tdnn_standard,
    small_tdnn_se,
)
from src.utils.paths import PROJECT_ROOT, resolve_path

SAMPLE_RATE = 16000
MIN_DURATION_SEC = 1.0
MIN_SAMPLES = int(SAMPLE_RATE * MIN_DURATION_SEC)

DEFAULT_CHECKPOINT_CANDIDATES = (
    "checkpoints/s0_seed42/best_dev_eer.pt",
    "checkpoints/pilot_s0_mac/best_dev_eer.pt",
)

EXAMPLE_PAIRS: Dict[str, Dict[str, str]] = {
    "same": {
        "id": "same",
        "title": "Same Speaker (Spk 7176, Cross-Chapter)",
        "expected_verdict": "SAME",
        "enrollment_path": "data/raw/LibriSpeech/test-clean/7176/88083/7176-88083-0008.flac",
        "test_path": "data/raw/LibriSpeech/test-clean/7176/92135/7176-92135-0037.flac",
    },
    "different": {
        "id": "different",
        "title": "Different Speakers (Spk 4970 vs Spk 4992)",
        "expected_verdict": "DIFFERENT",
        "enrollment_path": "data/raw/LibriSpeech/test-clean/4970/29095/4970-29095-0030.flac",
        "test_path": "data/raw/LibriSpeech/test-clean/4992/41797/4992-41797-0000.flac",
    },
}


def find_ffmpeg() -> Optional[str]:
    """Return path to ffmpeg executable if installed, else None."""
    return shutil.which("ffmpeg")


def decode_audio_bytes_in_memory(
    audio_bytes: bytes,
    label: str = "audio",
) -> Tuple[np.ndarray, float]:
    """Decode browser audio blob (webm/ogg/wav/flac) to 16 kHz mono float32 PCM in memory.

    Never writes audio to disk. Raises RuntimeError if ffmpeg is missing or ValueError
    if decoding fails or the clip is shorter than ~1 second.
    """
    ffmpeg_bin = find_ffmpeg()
    if not ffmpeg_bin:
        raise RuntimeError(
            "ffmpeg is not installed or not found on PATH. "
            "Please install ffmpeg (e.g. 'brew install ffmpeg') to decode audio."
        )

    if not audio_bytes:
        raise ValueError(f"Uploaded {label} clip is empty (0 bytes).")

    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=audio_bytes, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        err_msg = proc.stderr.decode("utf-8", errors="replace").strip() or "unknown ffmpeg error"
        raise ValueError(f"Failed to decode {label} clip with ffmpeg: {err_msg}")

    pcm = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    duration_sec = float(len(pcm)) / float(SAMPLE_RATE)

    if len(pcm) < MIN_SAMPLES:
        raise ValueError(
            f"{label.capitalize()} clip is too short ({duration_sec:.2f}s). "
            f"Minimum required duration is {MIN_DURATION_SEC:.1f}s."
        )

    return pcm, duration_sec


def resolve_checkpoint_path(checkpoint: Optional[Union[str, Path]] = None) -> Path:
    """Resolve explicit or default checkpoint path (s0_seed42 falling back to pilot_s0_mac)."""
    if checkpoint is not None:
        p = Path(checkpoint).expanduser()
        if p.is_absolute() and p.exists():
            return p.resolve()
        resolved = resolve_path(p)
        if resolved.exists():
            return resolved
        repo_candidate = (PROJECT_ROOT / p).resolve()
        if repo_candidate.exists():
            return repo_candidate
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

    for rel_candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        resolved = resolve_path(rel_candidate)
        if resolved.exists():
            return resolved
        repo_candidate = (PROJECT_ROOT / rel_candidate).resolve()
        if repo_candidate.exists():
            return repo_candidate

    raise FileNotFoundError(
        f"No default checkpoint found. Checked: {', '.join(DEFAULT_CHECKPOINT_CANDIDATES)}"
    )


def resolve_project_file(rel_or_abs: Union[str, Path]) -> Path:
    """Resolve a data/trial file against configured roots with fallback to PROJECT_ROOT."""
    resolved = resolve_path(rel_or_abs)
    if resolved.exists():
        return resolved
    fallback = (PROJECT_ROOT / rel_or_abs).resolve()
    if fallback.exists():
        return fallback
    return resolved


def instantiate_model_for_checkpoint(
    ckpt_payload: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, str]:
    """Instantiate model using the existing model factory matching the checkpoint config."""
    config_hash = ckpt_payload.get("config_hash", "")
    matched_cfg = None
    configs_dir = PROJECT_ROOT / "configs"
    if config_hash and configs_dir.exists():
        for cfg_path in sorted(configs_dir.glob("*.yaml")):
            if compute_config_hash(cfg_path) == config_hash:
                matched_cfg = load_config(cfg_path)
                break

    if matched_cfg is not None:
        model_name = str(matched_cfg.model.get("name", "ska_hnn"))
        emb_dim = int(matched_cfg.model.get("emb_dim", 192))
        in_features = int(matched_cfg.audio.get("n_mels", 80))
        if model_name == "ska_hnn":
            switch = str(matched_cfg.model.get("switch", "S0"))
            model = build_ska_hnn_from_config(matched_cfg).to(device)
            return model, f"ska_hnn ({switch})"
        if model_name == "small_tdnn_se":
            channels = int(matched_cfg.model.get("channels", 128))
            model = small_tdnn_se(in_features=in_features, channels=channels, emb_dim=emb_dim).to(device)
            return model, "small_tdnn_se"
        if model_name in {"ecapa_tdnn", "ecapa"}:
            variant = str(matched_cfg.model.get("variant", "param_matched")).lower()
            if variant == "standard":
                model = ecapa_tdnn_standard(in_features=in_features, emb_dim=emb_dim).to(device)
            else:
                channels = matched_cfg.model.get("channels", 400)
                channels = [int(c) for c in channels] if isinstance(channels, list) else int(channels)
                model = ECAPATDNN(in_features=in_features, channels=channels, emb_dim=emb_dim).to(device)
            return model, f"ecapa_tdnn ({variant})"

    # Fallback inspection of state_dict if config_hash is not matched
    state_dict = ckpt_payload.get("model_state_dict", ckpt_payload)
    if any(k.startswith("blocks.0.conv") for k in state_dict):
        model = small_tdnn_se(in_features=80, channels=128, emb_dim=192).to(device)
        return model, "small_tdnn_se"
    model = build_ska_hnn(switch="S0").to(device)
    return model, "ska_hnn (S0)"


def get_threshold_cache_path(ckpt_path: Path) -> Path:
    """Return path to the threshold JSON cache file stored next to the checkpoint."""
    return ckpt_path.with_suffix(".threshold.json")


def compute_or_load_threshold(
    model: torch.nn.Module,
    ckpt_path: Path,
    ckpt_payload: Dict[str, Any],
    device: torch.device,
    trials_path: Optional[Path] = None,
) -> Tuple[float, float, int, Path]:
    """Load cached dev-clean EER threshold next to checkpoint, or recompute if missing."""
    cache_path = get_threshold_cache_path(ckpt_path)
    epoch = int(ckpt_payload.get("epoch", 0))
    stored_dev_eer = ckpt_payload.get("best_dev_eer")

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if "threshold" in cached:
                threshold = float(cached["threshold"])
                dev_eer = (
                    round(float(stored_dev_eer), 2)
                    if stored_dev_eer is not None and np.isfinite(float(stored_dev_eer))
                    else round(float(cached.get("dev_eer", cached.get("computed_dev_eer_pct", 0.0))), 2)
                )
                return threshold, dev_eer, epoch, cache_path
        except Exception:
            pass

    if trials_path is None:
        trials_path = resolve_project_file("data/trials/trials_dev-clean.txt")

    eval_res = evaluate_trials(
        model=model,
        trials_path=trials_path,
        trials_meta_path=None,
        device=device,
        show_progress=True,
    )
    threshold = float(eval_res["eer_threshold"])
    computed_eer_pct = float(eval_res["eer_pct"])
    dev_eer = (
        round(float(stored_dev_eer), 2)
        if stored_dev_eer is not None and np.isfinite(float(stored_dev_eer))
        else computed_eer_pct
    )

    cache_payload = {
        "checkpoint_path": str(ckpt_path),
        "epoch": epoch,
        "dev_eer": dev_eer,
        "computed_dev_eer_pct": computed_eer_pct,
        "threshold": threshold,
        "eer_threshold": threshold,
        "trials_path": str(trials_path),
    }
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_payload, f, indent=2)
    except OSError:
        pass

    return threshold, dev_eer, epoch, cache_path


class ServerState:
    """Holds loaded model, full-utterance extractor, and checkpoint metadata."""

    def __init__(self) -> None:
        self.extractor: Optional[EmbeddingExtractor] = None
        self.model: Optional[torch.nn.Module] = None
        self.model_name: str = ""
        self.parameter_count: int = 0
        self.checkpoint_path: str = ""
        self.checkpoint_abs_path: Optional[Path] = None
        self.epoch: int = 0
        self.dev_eer: float = 0.0
        self.threshold: float = 0.0
        self.threshold_cache_path: str = ""
        self.device: torch.device = torch.device("cpu")

    def initialize(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        device_str: str = "cpu",
        trials_path: Optional[Path] = None,
    ) -> None:
        """Load checkpoint, instantiate extractor, and load or compute dev-clean EER threshold."""
        self.device = torch.device(device_str)
        ckpt_path = resolve_checkpoint_path(checkpoint)

        try:
            payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(ckpt_path, map_location=self.device)

        model, model_name = instantiate_model_for_checkpoint(payload, self.device)
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(state_dict)
        model.eval()

        # Measure parameter count via torchinfo
        summary = torchinfo.summary(model, verbose=0)
        param_count = int(summary.total_params)

        extractor = EmbeddingExtractor(model=model, device=self.device, batch_size=1)
        threshold, dev_eer, epoch, cache_path = compute_or_load_threshold(
            model=model,
            ckpt_path=ckpt_path,
            ckpt_payload=payload if isinstance(payload, dict) else {},
            device=self.device,
            trials_path=trials_path,
        )

        try:
            display_ckpt_path = str(ckpt_path.relative_to(PROJECT_ROOT))
        except ValueError:
            display_ckpt_path = str(ckpt_path)

        self.model = model
        self.extractor = extractor
        self.model_name = model_name
        self.parameter_count = param_count
        self.checkpoint_path = display_ckpt_path
        self.checkpoint_abs_path = ckpt_path
        self.epoch = epoch
        self.dev_eer = dev_eer
        self.threshold = threshold
        self.threshold_cache_path = str(cache_path)

        print(
            f"[Neurovox Startup] Checkpoint: {display_ckpt_path} | "
            f"Epoch: {self.epoch} | Dev EER: {self.dev_eer:.2f}% | "
            f"Threshold: {self.threshold:.4f} | Device: {self.device}",
            flush=True,
        )


STATE = ServerState()


def ensure_state_loaded() -> ServerState:
    """Ensure the global server state is initialized."""
    if STATE.extractor is None:
        STATE.initialize()
    return STATE


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize model and threshold at FastAPI startup if not already initialized."""
    ensure_state_loaded()
    yield


app = FastAPI(
    title="Neurovox Speaker Verification Demo",
    version="0.1.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the main single-page frontend."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/info")
async def get_info() -> Dict[str, Any]:
    """Return loaded model metadata, parameter count, checkpoint path/epoch, dev EER, and threshold."""
    state = ensure_state_loaded()
    return {
        "model_name": state.model_name,
        "parameter_count": state.parameter_count,
        "checkpoint_path": state.checkpoint_path,
        "checkpoint_epoch": state.epoch,
        "epoch": state.epoch,
        "dev_eer": state.dev_eer,
        "dev_eer_pct": state.dev_eer,
        "threshold": state.threshold,
        "device": str(state.device),
        "examples": list(EXAMPLE_PAIRS.values()),
    }


@app.get("/api/examples")
async def list_examples() -> Dict[str, Any]:
    """Return local LibriSpeech test-clean example pairs metadata."""
    return {"examples": list(EXAMPLE_PAIRS.values())}


@app.get("/api/examples/{pair_id}/{slot}")
async def get_example_audio(pair_id: str, slot: str) -> Response:
    """Serve local LibriSpeech test-clean example audio file bytes from disk into memory."""
    if pair_id not in EXAMPLE_PAIRS:
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown example pair '{pair_id}'.", "detail": f"Unknown example pair '{pair_id}'."},
        )
    if slot not in {"enrollment", "test"}:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid slot '{slot}'. Expected 'enrollment' or 'test'.", "detail": f"Invalid slot '{slot}'."},
        )

    rel_path = EXAMPLE_PAIRS[pair_id][f"{slot}_path"]
    file_path = resolve_project_file(rel_path)
    if not file_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Example file not found: {rel_path}", "detail": f"Example file not found: {rel_path}"},
        )

    raw_bytes = file_path.read_bytes()
    return Response(
        content=raw_bytes,
        media_type="audio/flac",
        headers={"Content-Disposition": f'inline; filename="{file_path.name}"'},
    )


@app.post("/api/verify")
async def verify_speakers(
    enrollment: Optional[UploadFile] = File(default=None),
    test: Optional[UploadFile] = File(default=None),
    enroll: Optional[UploadFile] = File(default=None),
    audio1: Optional[UploadFile] = File(default=None),
    audio2: Optional[UploadFile] = File(default=None),
) -> JSONResponse:
    """Verify whether two uploaded/recorded audio clips belong to the same speaker."""
    state = ensure_state_loaded()

    enroll_upload = enrollment or enroll or audio1
    test_upload = test or audio2

    if enroll_upload is None or test_upload is None:
        msg = "Both 'enrollment' and 'test' audio blobs are required."
        return JSONResponse(status_code=400, content={"error": msg, "detail": msg})

    try:
        enroll_bytes = await enroll_upload.read()
        test_bytes = await test_upload.read()

        pcm_enroll, dur_enroll = decode_audio_bytes_in_memory(enroll_bytes, label="enrollment")
        pcm_test, dur_test = decode_audio_bytes_in_memory(test_bytes, label="test")
    except RuntimeError as exc:
        msg = str(exc)
        return JSONResponse(status_code=500, content={"error": msg, "detail": msg})
    except ValueError as exc:
        msg = str(exc)
        return JSONResponse(status_code=400, content={"error": msg, "detail": msg})

    assert state.extractor is not None
    emb_enroll = state.extractor.extract_from_waveform(torch.from_numpy(pcm_enroll))
    emb_test = state.extractor.extract_from_waveform(torch.from_numpy(pcm_test))

    cosine_score = float(np.clip(np.dot(emb_enroll, emb_test), -1.0, 1.0))
    threshold = float(state.threshold)
    verdict = "SAME" if cosine_score >= threshold else "DIFFERENT"

    return JSONResponse(
        status_code=200,
        content={
            "score": round(cosine_score, 4),
            "cosine_score": round(cosine_score, 4),
            "threshold": round(threshold, 4),
            "verdict": verdict,
            "enrollment_duration": round(dur_enroll, 3),
            "test_duration": round(dur_test, 3),
            "enrollment_duration_sec": round(dur_enroll, 3),
            "test_duration_sec": round(dur_test, 3),
            "durations": {
                "enrollment": round(dur_enroll, 3),
                "test": round(dur_test, 3),
            },
            "checkpoint_epoch": state.epoch,
            "dev_eer": state.dev_eer,
        },
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for starting the Neurovox verification server."""
    parser = argparse.ArgumentParser(description="Neurovox Speaker Verification Demo Server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to model checkpoint (.pt). Defaults to checkpoints/s0_seed42/best_dev_eer.pt, "
            "falling back to checkpoints/pilot_s0_mac/best_dev_eer.pt."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device (default: 'cpu' so MPS training is undisturbed).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Entrypoint for `python app/server.py`."""
    args = parse_args(argv)
    STATE.initialize(checkpoint=args.checkpoint, device_str=args.device)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


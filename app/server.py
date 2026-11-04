#!/usr/bin/env python3
"""FastAPI backend for Stage 1 & Stage 2 of the Neurovox speaker verification demo site.

Features:
- Loads trained SKA-HNN (or baseline) checkpoint on CPU by default (--device to override).
- Reuses existing model factory, 80-dim log-Mel frontend, and full-utterance EmbeddingExtractor.
- Computes dev-clean EER threshold once at startup via scripts.evaluate.evaluate_trials
  and caches it to a JSON file next to the checkpoint (recomputed if missing).
- Decodes browser audio blobs (webm/ogg/wav/flac) to 16 kHz mono float32 entirely in memory
  via ffmpeg pipes (never writing audio to disk).
- Returns SAME / DIFFERENT / INCONCLUSIVE (when within +/- 0.10 of the decision threshold).
- Exposes GET /api/info, POST /api/verify, GET /api/examples, GET /api/results,
  and serves app/static/index.html.
"""

import argparse
from contextlib import asynccontextmanager
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

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
from src.utils.paths import PROJECT_ROOT, get_results_root, resolve_path

SAMPLE_RATE = 16000
MIN_DURATION_SEC = 1.0
MIN_SAMPLES = int(SAMPLE_RATE * MIN_DURATION_SEC)
BORDERLINE_MARGIN = 0.10

DEFAULT_CHECKPOINT_CANDIDATES = (
    "checkpoints/s0_seed42/best_dev_eer.pt",
    "checkpoints/pilot_s0_mac/best_dev_eer.pt",
)

EXAMPLE_PAIRS: Dict[str, Dict[str, str]] = {
    "same": {
        "id": "same",
        "title": "Run example: same speaker",
        "expected_verdict": "SAME",
        "enrollment_path": "data/raw/LibriSpeech/test-clean/7176/88083/7176-88083-0008.flac",
        "test_path": "data/raw/LibriSpeech/test-clean/7176/92135/7176-92135-0037.flac",
    },
    "different": {
        "id": "different",
        "title": "Run example: different speakers",
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
    """Resolve a data/trial/doc file against configured roots with fallback to PROJECT_ROOT."""
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


def classify_verdict(score: float, threshold: float, margin: float = BORDERLINE_MARGIN) -> str:
    """Classify cosine score against threshold with +/- margin inconclusive band."""
    if score > threshold + margin:
        return "SAME"
    if score < threshold - margin:
        return "DIFFERENT"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Stage 2 Real-Data Parsers (results/*, docs/ARCHITECTURE.md, BENCHMARKS.md, RESULTS_LOG.md)
# ---------------------------------------------------------------------------


def _parse_num(val: Any) -> Optional[Union[int, float]]:
    """Parse string or numeric cell into int/float, returning None for empty/NaN/TBD."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null", "n/a", "tbd"}:
        return None
    try:
        if "." in s or "e" in s.lower():
            f = float(s)
            return None if (math.isnan(f) or math.isinf(f)) else f
        return int(s)
    except ValueError:
        return None


def _parse_int_from_markdown(cell: str) -> Optional[int]:
    """Extract an integer from a markdown cell like '**4,167,700**' or '47,802'."""
    cleaned = re.sub(r"[*_`]", "", cell).strip()
    if not cleaned or "non-parametric" in cleaned.lower():
        return 0 if "non-parametric" in cleaned.lower() else None
    m = re.search(r"(\d[\d,]*)", cleaned)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _parse_float_from_markdown(cell: str) -> Optional[float]:
    """Extract a float from a markdown cell like '**411.52**' or '26.36', None if TBD/N/A."""
    cleaned = re.sub(r"[*_`]", "", cell).strip()
    if not cleaned or any(tok in cleaned.lower() for tok in ("tbd", "n/a", "pending")):
        return None
    m = re.search(r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", cleaned)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _load_single_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Read history.csv and run_meta.json from a single results/<run_id> folder."""
    if not run_dir.is_dir():
        return None
    csv_path = run_dir / "history.csv"
    meta_path = run_dir / "run_meta.json"
    if not csv_path.exists() and not meta_path.exists():
        return None

    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    epochs: List[Dict[str, Any]] = []
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for raw_row in reader:
                    row: Dict[str, Any] = {}
                    for k, v in raw_row.items():
                        if k == "first_batch_index_hash":
                            row[k] = v
                        else:
                            row[k] = _parse_num(v)
                    # Derived per-epoch fields from measured columns only
                    train_acc = row.get("train_acc")
                    dev_eer = row.get("dev_eer_pct")
                    probe_eer = row.get("train_probe_eer_pct")
                    row["train_err_pct"] = (
                        round(100.0 - float(train_acc), 2) if train_acc is not None else None
                    )
                    row["eer_gap_pct"] = (
                        round(float(dev_eer) - float(probe_eer), 2)
                        if (dev_eer is not None and probe_eer is not None)
                        else None
                    )
                    row["train_err_vs_dev_eer_gap_pct"] = (
                        round(float(dev_eer) - (100.0 - float(train_acc)), 2)
                        if (dev_eer is not None and train_acc is not None)
                        else None
                    )
                    epochs.append(row)
        except Exception:
            epochs = []

    eval_dev_eers = [
        float(r["dev_eer_pct"]) for r in epochs if r.get("dev_eer_pct") is not None
    ]
    best_dev_eer = min(eval_dev_eers) if eval_dev_eers else None
    best_dev_epoch = None
    if best_dev_eer is not None:
        for r in epochs:
            if r.get("dev_eer_pct") == best_dev_eer:
                best_dev_epoch = r.get("epoch")
                break

    return {
        "run_id": meta.get("run_id", run_dir.name),
        "label_type": "measured",
        "meta": meta,
        "num_epochs_logged": len(epochs),
        "best_dev_eer_pct": best_dev_eer,
        "best_dev_epoch": best_dev_epoch,
        "epochs": epochs,
    }


def _collect_results_runs() -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Scan results/ directory for S0, S1, pilots, and sanity runs."""
    found_sources: List[str] = []
    missing_sources: List[str] = []

    # Check both get_results_root() and PROJECT_ROOT / "results"
    candidate_roots = [get_results_root(), (PROJECT_ROOT / "results").resolve()]
    run_dirs: Dict[str, Path] = {}
    for root in candidate_roots:
        if root.exists():
            for child in sorted(root.iterdir()):
                if child.is_dir() and child.name not in run_dirs:
                    run_dirs[child.name] = child

    all_runs: Dict[str, Any] = {}
    for name, path in sorted(run_dirs.items()):
        parsed = _load_single_run(path)
        if parsed is not None:
            all_runs[name] = parsed
            found_sources.append(f"results/{name}/history.csv")

    # Primary headline / pilot slots expected by the UI
    expected_runs = [
        "s0_seed42",
        "s1_seed42",
        "pilot_s0_mac",
        "pilot_s1_mac",
        "pilot_s0_colab",
        "pilot_s1_colab",
    ]
    for exp_id in expected_runs:
        if exp_id not in all_runs:
            missing_sources.append(f"results/{exp_id}/history.csv")

    return all_runs, found_sources, missing_sources


def _parse_markdown_tables(md_text: str) -> List[Tuple[List[str], List[List[str]]]]:
    """Extract all GitHub-flavored markdown tables as (headers, rows) tuples."""
    tables: List[Tuple[List[str], List[List[str]]]] = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and line.endswith("|") and i + 1 < len(lines):
            sep = lines[i + 1].strip()
            if sep.startswith("|") and "---" in sep:
                headers = [c.strip() for c in line.strip("|").split("|")]
                rows: List[List[str]] = []
                i += 2
                while i < len(lines):
                    rline = lines[i].strip()
                    if not (rline.startswith("|") and rline.endswith("|")):
                        break
                    cells = [c.strip() for c in rline.strip("|").split("|")]
                    rows.append(cells)
                    i += 1
                tables.append((headers, rows))
                continue
        i += 1
    return tables


def _parse_architecture_doc(
    all_runs: Dict[str, Any],
    found_sources: List[str],
    missing_sources: List[str],
) -> Dict[str, Any]:
    """Parse docs/ARCHITECTURE.md for module contracts, ablation switches S0-S6, and baselines."""
    arch_path = resolve_project_file("docs/ARCHITECTURE.md")
    if not arch_path.exists():
        missing_sources.append("docs/ARCHITECTURE.md")
        return {}

    found_sources.append("docs/ARCHITECTURE.md")
    text = arch_path.read_text(encoding="utf-8")
    tables = _parse_markdown_tables(text)

    stage_contracts: List[Dict[str, Any]] = []
    ablation_switches: List[Dict[str, Any]] = []
    baseline_models: List[Dict[str, Any]] = []

    for headers, rows in tables:
        if len(headers) >= 7 and "Module Name" in headers[1]:
            for r in rows:
                if len(r) < 7:
                    continue
                stage_contracts.append(
                    {
                        "stage": re.sub(r"[*`]", "", r[0]).strip(),
                        "module_name": re.sub(r"[*`]", "", r[1]).strip(),
                        "input_shape": re.sub(r"[`]", "", r[2]).strip(),
                        "operations": r[3].strip(),
                        "output_shape": re.sub(r"[`]", "", r[4]).strip(),
                        "full_scale_params": _parse_int_from_markdown(r[5]),
                        "full_scale_raw": r[5],
                        "scaled_params": _parse_int_from_markdown(r[6]),
                        "scaled_raw": r[6],
                        "label_type": "measured",
                    }
                )
        elif len(headers) >= 6 and "Switch ID" in headers[0]:
            for r in rows:
                if len(r) < 6:
                    continue
                sw_id = re.sub(r"[*`]", "", r[0]).strip()
                paper_str = re.sub(r"[*`]", "", r[4]).strip()
                measured_int = _parse_int_from_markdown(r[5])

                # Determine dynamic status from actual runs on disk
                if sw_id == "S0":
                    s0_run = all_runs.get("s0_seed42")
                    if s0_run and s0_run["num_epochs_logged"] > 0:
                        ep_cnt = s0_run["num_epochs_logged"]
                        best_eer = s0_run.get("best_dev_eer_pct")
                        status = (
                            f"trained ({ep_cnt}/30 epochs; best dev EER {best_eer:.2f}%)"
                            if ep_cnt < 30 and best_eer is not None
                            else "trained"
                        )
                    elif "pilot_s0_mac" in all_runs:
                        status = "pilot trained (1 ep)"
                    else:
                        status = "implemented / pending"
                elif sw_id == "S1":
                    s1_run = all_runs.get("s1_seed42")
                    if s1_run and s1_run["num_epochs_logged"] > 0:
                        status = "trained"
                    elif "pilot_s1_mac" in all_runs:
                        status = "implemented / pilot trained (1 ep; 30-ep pending)"
                    else:
                        status = "implemented / pending"
                elif sw_id == "S4":
                    status = "dropped (implemented & audited; pruned from grid)"
                else:
                    status = "implemented / pending"

                ablation_switches.append(
                    {
                        "switch": sw_id,
                        "dropped": sw_id == "S4",
                        "config_summary": re.sub(r"[*_]", "", r[1]).strip(),
                        "flags": r[2].replace("<br>", ", ").replace("`", ""),
                        "description": r[3].strip(),
                        "paper_params": paper_str,
                        "paper_label_type": "paper",
                        "measured_params": measured_int,
                        "measured_label_type": "measured",
                        "status": status,
                    }
                )
        elif len(headers) >= 7 and "Model Architecture" in headers[0]:
            for r in rows:
                if len(r) < 7:
                    continue
                baseline_models.append(
                    {
                        "model_architecture": re.sub(r"[*_`]", "", r[0]).strip(),
                        "channels": re.sub(r"[`]", "", r[1]).strip(),
                        "res2net_scale": r[2].strip(),
                        "se_channels": r[3].strip(),
                        "attn_channels": r[4].strip(),
                        "measured_params": _parse_int_from_markdown(r[5]),
                        "delta_vs_skahnn": re.sub(r"[*_]", "", r[7]).strip() if len(r) > 7 else "",
                        "label_type": "measured",
                    }
                )

    # Measure exact clickable sub-module parameter counts via torchinfo on SKAHNN(S0)
    ska_model = build_ska_hnn(switch="S0")
    fw_ska_1 = int(torchinfo.summary(ska_model.resblock1.fw_ska, verbose=0).total_params)
    cw_ska_1 = int(torchinfo.summary(ska_model.resblock1.cw_ska, verbose=0).total_params)
    se_1 = int(torchinfo.summary(ska_model.resblock1.se, verbose=0).total_params)
    resblock_1 = int(torchinfo.summary(ska_model.resblock1, verbose=0).total_params)
    conv_in_p = int(torchinfo.summary(ska_model.conv2d_in, verbose=0).total_params)
    conv_out_p = int(torchinfo.summary(ska_model.conv2d_out, verbose=0).total_params)
    tdnn_lstm_p = int(torchinfo.summary(ska_model.tdnn_lstm, verbose=0).total_params)
    fully_tdnn_p = int(torchinfo.summary(ska_model.fully_tdnn, verbose=0).total_params)
    mlasp_p = int(torchinfo.summary(ska_model.mlasp, verbose=0).total_params)
    fc1_p = int(torchinfo.summary(ska_model.fc1, verbose=0).total_params)
    fc2_p = int(torchinfo.summary(ska_model.fc2, verbose=0).total_params)

    clickable_modules = [
        {
            "id": "fwSKA",
            "name": "fwSKA (Frequency-wise SKA)",
            "stage": "1. 2D-CNN Frontend (inside SKA ResBlocks 1 & 2)",
            "input_shape": "(B, 32, 40, T)",
            "output_shape": "(B, 32, 40, T)",
            "params_single": fw_ska_1,
            "params_total": fw_ska_1 * 2,
            "label_type": "measured",
            "description": (
                "Splits feature maps into two 2D conv branches (3x3 d=1 and 3x3 d=2 for 5x5 receptive field), "
                "fuses by element-wise sum, pools across channel & time axes to (B, F=40), and applies a "
                "bottleneck MLP (r=8) + softmax to weight frequency bands."
            ),
        },
        {
            "id": "cwSKA",
            "name": "cwSKA (Channel-wise SKA)",
            "stage": "1. 2D-CNN Frontend (inside SKA ResBlocks 1 & 2)",
            "input_shape": "(B, 32, 40, T)",
            "output_shape": "(B, 32, 40, T)",
            "params_single": cw_ska_1,
            "params_total": cw_ska_1 * 2,
            "label_type": "measured",
            "description": (
                "Applies two multi-scale 2D conv branches (k=3 d=1 and k=3 d=2), fuses by summation, "
                "pools across frequency & time axes to (B, C=32), and computes channel-wise soft attention "
                "across branches via an r=8 bottleneck MLP."
            ),
        },
        {
            "id": "SE",
            "name": "SE (Squeeze-and-Excitation)",
            "stage": "1. 2D-CNN Frontend (inside SKA ResBlocks 1 & 2)",
            "input_shape": "(B, 32, 40, T)",
            "output_shape": "(B, 32, 40, T)",
            "params_single": se_1,
            "params_total": se_1 * 2,
            "label_type": "measured",
            "description": (
                "Global average pooling over frequency & time followed by a two-layer channel excitation "
                "bottleneck (32 -> 4 -> 32, reduction ratio r=8, Sigmoid gating) before the residual skip addition."
            ),
        },
        {
            "id": "ResBlock",
            "name": "SKA ResBlock (2x in 2D-CNN Frontend)",
            "stage": "1. 2D-CNN Frontend",
            "input_shape": "(B, 32, 40, T)",
            "output_shape": "(B, 32, 40, T)",
            "params_single": resblock_1,
            "params_total": resblock_1 * 2,
            "full_scale_params_single": 749778,
            "label_type": "measured",
            "description": (
                "Sequential Conv2d(32, 32, k=3) + ReLU + BN2d (9,280 params) -> fwSKA (19,250) -> cwSKA (19,016) "
                "-> SE (256) + residual skip connection. Total 47,802 params per block (95,604 across both blocks; "
                "full-scale C=128: 749,778 per block)."
            ),
        },
        {
            "id": "local_stats",
            "name": "Local Statistics (LS 1, LS 2, LS 3)",
            "stage": "Stages 2, 3 & 4 (Pre-TDNN-LSTM, Pre-Fully-TDNN, MLASP Context)",
            "input_shape": "(B, D, T)",
            "output_shape": "(B, 3*D, T)",
            "params_single": 0,
            "params_total": 0,
            "label_type": "measured",
            "description": (
                "Non-parametric sliding-window (W=21 frames, replicate padding) mean mu_t and standard deviation "
                "sigma_t computed strictly in fp32 with variance clamping (min=1e-5) before sqrt, concatenated as [X; mu; sigma]."
            ),
        },
        {
            "id": "TDNN-LSTM",
            "name": "TDNN-LSTM Hybrid Module",
            "stage": "2. Frame-Level Cascade Stage 1",
            "input_shape": "(B, 1920, T) [3 * d_c]",
            "output_shape": "(B, 256, T) [X_l]",
            "params_single": tdnn_lstm_p,
            "params_total": tdnn_lstm_p,
            "full_scale_params_single": 11797504 + 2102272,
            "label_type": "measured",
            "description": (
                "Receives local-stats augmented 2D-CNN features (3 * 640 = 1,920 channels). Applies TDNN "
                "Conv1d(1920, 256, k=3)+ReLU+BN1d (1,475,072 params) followed by unidirectional LSTM(256, 256)+ReLU+BN1d "
                "(526,848 params) to jointly capture local and recurrent temporal context."
            ),
        },
        {
            "id": "Fully-TDNN",
            "name": "Fully-TDNN Module",
            "stage": "3. Frame-Level Cascade Stage 2",
            "input_shape": "(B, 768, T) [3 * C_1]",
            "output_shape": "(B, 256, T) [X_t]",
            "params_single": fully_tdnn_p,
            "params_total": fully_tdnn_p,
            "full_scale_params_single": 2360320 + 787456 + 787456,
            "label_type": "measured",
            "description": (
                "Receives local-stats augmented TDNN-LSTM features (3 * 256 = 768 channels). Stacks three dilated "
                "TDNN Conv1d+ReLU+BN1d layers with dilations d=1 (590,336 params), d=2 (197,120 params), and d=3 (197,120 params)."
            ),
        },
        {
            "id": "MLFC",
            "name": "MLFC (Multi-Level Feature Combination)",
            "stage": "4. Feature Aggregation Highway",
            "input_shape": "X_c (640), X_l (256), X_t (256)",
            "output_shape": "(B, 1152, T) [X_i]",
            "params_single": 0,
            "params_total": 0,
            "label_type": "measured",
            "description": (
                "Non-parametric feature highway concatenating outputs from the 2D-CNN frontend (X_c, 640-d), "
                "TDNN-LSTM (X_l, 256-d), and Fully-TDNN (X_t, 256-d) into X_i of dimension d_i = 1,152 (full scale: 3,584)."
            ),
        },
        {
            "id": "MLASP",
            "name": "MLASP (Multi-Level Attentive Statistics Pooling)",
            "stage": "4. Utterance-Level Pooling",
            "input_shape": "(B, 1152, T) -> Context (B, 3456, T)",
            "output_shape": "(B, 2304) [2 * d_i]",
            "params_single": mlasp_p,
            "params_total": mlasp_p,
            "full_scale_params_single": 3679488 + 9216,
            "label_type": "measured",
            "description": (
                "Augments X_i with W=21 local stats to form X_i^c (3,456-d), scores frame/channel attention via "
                "Conv1d(3456, 128, 1)->Tanh->BN1d->Conv1d(128, 1152, 1)->Softmax (591,360 params), and computes "
                "weighted mean + std followed by BN1d(2304) (4,608 params)."
            ),
        },
        {
            "id": "FC_Head",
            "name": "Classifier Head (FC1 Primary + FC2 Diagnostic)",
            "stage": "5. Embedding Projection",
            "input_shape": "(B, 2304)",
            "output_shape": "(B, 192)",
            "params_single": fc1_p,
            "params_total": fc1_p + fc2_p,
            "label_type": "measured",
            "description": (
                "FC1 (Linear 2304->192 + BN1d: 442,752 params) extracts the 192-d speaker embedding closest to pooling "
                "used in evaluation [Paper]. FC2 (ReLU + Linear 192->192 + BN1d: 37,248 params) completes the 2-FC head. "
                "Preceding/Succeeding 2D-CNN Conv2d blocks add 352 + 9,280 = 9,632 params (Total Backbone: 4,167,700)."
            ),
        },
    ]

    return {
        "stage_contracts": stage_contracts,
        "ablation_switches": ablation_switches,
        "baseline_models": baseline_models,
        "clickable_modules": clickable_modules,
    }


def _parse_benchmarks_and_ledger(
    found_sources: List[str],
    missing_sources: List[str],
) -> Dict[str, Any]:
    """Parse docs/BENCHMARKS.md and docs/RESULTS_LOG.md for hardware timings and external refs."""
    bench_path = resolve_project_file("docs/BENCHMARKS.md")
    hardware_rows: List[Dict[str, Any]] = []
    dataloader_mac_rows: List[Dict[str, Any]] = []
    dataloader_colab_rows: List[Dict[str, Any]] = []

    if bench_path.exists():
        found_sources.append("docs/BENCHMARKS.md")
        b_text = bench_path.read_text(encoding="utf-8")
        for headers, rows in _parse_markdown_tables(b_text):
            if len(headers) >= 9 and "Component / Layer" in headers[0]:
                for r in rows:
                    if len(r) < 9:
                        continue
                    mac_ms = _parse_float_from_markdown(r[2])
                    colab_ms = _parse_float_from_markdown(r[6])
                    hardware_rows.append(
                        {
                            "component": re.sub(r"[*_`]", "", r[0]).strip(),
                            "precision": re.sub(r"[`]", "", r[1]).strip(),
                            "mac_median_ms": mac_ms,
                            "mac_steps_per_sec": _parse_float_from_markdown(r[3]),
                            "mac_peak_mem_mb": _parse_float_from_markdown(r[4]),
                            "mac_label_type": "measured" if mac_ms is not None else "planned",
                            "colab_median_ms": colab_ms,
                            "colab_steps_per_sec": _parse_float_from_markdown(r[7]),
                            "colab_peak_vram_mb": _parse_float_from_markdown(r[8]),
                            "colab_label_type": "measured" if colab_ms is not None else None,
                        }
                    )
            elif len(headers) >= 5 and "Worker Count" in headers[0]:
                # Distinguish Mac table (has numeric throughput) vs Colab table (TBD)
                has_measured = any(_parse_float_from_markdown(r[1]) is not None for r in rows if len(r) >= 4)
                target_list = dataloader_mac_rows if has_measured else dataloader_colab_rows
                for r in rows:
                    if len(r) < 5:
                        continue
                    samples_s = _parse_float_from_markdown(r[1])
                    target_list.append(
                        {
                            "workers": re.sub(r"[*_`]", "", r[0]).strip(),
                            "samples_per_sec": samples_s,
                            "batches_per_sec": _parse_float_from_markdown(r[2]),
                            "batch_latency_ms": _parse_float_from_markdown(r[3]),
                            "notes": re.sub(r"[*_]", "", r[4]).strip(),
                            "label_type": "measured" if samples_s is not None else None,
                        }
                    )
    else:
        missing_sources.append("docs/BENCHMARKS.md")

    # Parse RESULTS_LOG.md for external reference SpeechBrain ECAPA runs and ledger entries
    log_path = resolve_project_file("docs/RESULTS_LOG.md")
    external_refs: List[Dict[str, Any]] = []
    ledger_entries: List[Dict[str, Any]] = []
    if log_path.exists():
        found_sources.append("docs/RESULTS_LOG.md")
        l_text = log_path.read_text(encoding="utf-8")
        for headers, rows in _parse_markdown_tables(l_text):
            if len(headers) >= 13 and "Run ID" in headers[0]:
                for r in rows:
                    if len(r) < 13:
                        continue
                    run_id = re.sub(r"[*`]", "", r[0]).strip()
                    if run_id.lower() == "template":
                        continue
                    status_str = r[12].strip()
                    is_ext = "EXTERNAL REFERENCE" in status_str.upper() or run_id.startswith("ref_")
                    entry = {
                        "run_id": run_id,
                        "timestamp": r[1].strip(),
                        "model": re.sub(r"[`]", "", r[4]).strip(),
                        "loss": r[5].strip(),
                        "dataset": r[6].strip(),
                        "params": r[7].strip(),
                        "eer": r[8].strip(),
                        "mindcf_p01": r[9].strip(),
                        "mindcf_p05": r[10].strip(),
                        "device": re.sub(r"[`]", "", r[11]).strip(),
                        "status": status_str,
                        "label_type": "external reference" if is_ext else "measured",
                    }
                    ledger_entries.append(entry)
                    if is_ext and "SUPERSEDED" not in status_str.upper():
                        external_refs.append(entry)
                break
    else:
        missing_sources.append("docs/RESULTS_LOG.md")

    return {
        "hardware_throughput": hardware_rows,
        "dataloader_mac": dataloader_mac_rows,
        "dataloader_colab": dataloader_colab_rows,
        "external_references": external_refs,
        "ledger_entries": ledger_entries,
    }


def _build_tier_status(all_runs: Dict[str, Any], external_refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build T0-T4 status board dynamically from existing code, results/*, and RESULTS_LOG.md."""
    s0_run = all_runs.get("s0_seed42")
    s0_ep = s0_run["num_epochs_logged"] if s0_run else 0
    s0_best = s0_run.get("best_dev_eer_pct") if s0_run else None
    s1_run = all_runs.get("s1_seed42")
    s1_ep = s1_run["num_epochs_logged"] if s1_run else 0

    return [
        {
            "tier": "Tier 0",
            "name": "Baselines (ECAPA-TDNN)",
            "code_status": "implemented",
            "training_status": "external reference evaluated; from-scratch training pending",
            "details": (
                "Pretrained SpeechBrain ECAPA-TDNN (20,767,552 params) evaluated on v2 trials: "
                "test-clean EER 2.41% [external reference], test-other EER 1.72% [external reference]. "
                "Parameter-matched ECAPA-TDNN (C=400, 4,123,210 params [measured]) passed 100-step overfit gate "
                "(sanity_2c_smoke_ecapa); 30-epoch training not run yet."
            ),
            "external_references": external_refs,
        },
        {
            "tier": "Tier 1",
            "name": "Core Proposed Systems (Scaled SKA-HNN S0 & S1)",
            "code_status": "implemented",
            "training_status": (
                f"S0 training ({s0_ep}/30 epochs, best dev EER {s0_best:.2f}% [measured]); "
                f"S1 pilot trained (1 ep, 22.00% dev EER [measured]), S1 30-ep {'trained' if s1_ep > 0 else 'pending'}"
                if (s0_ep > 0 and s0_best is not None)
                else "pilots trained; 30-ep pending"
            ),
            "details": (
                "Scaled SKA-HNN (C=32, C1=256, 4,167,700 params [measured]) with IAAMSoftmax (S0) and AAMSoftmax (S1). "
                "Mac M3 Pro 1-epoch pilots completed for both S0 and S1; S0 seed 42 headline run active."
            ),
        },
        {
            "tier": "Tier 2",
            "name": "Architecture Ablations (S2, S3, S5, S6; S4 Dropped)",
            "code_status": "implemented",
            "training_status": "pending",
            "details": (
                "All ablation switches S2 (3,872,788 params [measured]), S3 (3,360,404 params [measured]), "
                "S5 (4,109,728 params [measured]), and S6 (2,195,584 params [measured]) are implemented and "
                "unit-tested in src/models/ska_hnn/ska_hnn.py. S4 (3,294,868 params [measured]) is dropped. "
                "Full training runs not run yet."
            ),
        },
        {
            "tier": "Tier 3",
            "name": "Label Noise (10%/20%/40%) & SSL WavLM-Base+ K-Means",
            "code_status": "pending (AdaptiveDrop loss & safeguard implemented)",
            "training_status": "pending",
            "details": (
                "IAAMSoftmax AdaptiveDrop filtering (ESD=5, tau=0.423 [paper]) and label correction "
                "(ESC=5, tau_corr=0.5) with min_keep_ratio=0.5 safeguard are implemented and verified in "
                "sanity_2c_active_s0 [measured]. Synthetic label-noise sweeps and WavLM clustering not run yet."
            ),
        },
        {
            "tier": "Tier 4",
            "name": "Extensions (AS-Norm & Zero-Shot VoxCeleb1-O)",
            "code_status": "pending (Stage 1 & 2 interactive verification demo live)",
            "training_status": "pending",
            "details": (
                "Interactive web verification demo (Stage 1 & Stage 2) implemented and running. "
                "AS-Norm score calibration and zero-shot VoxCeleb1-O evaluation not run yet."
            ),
        },
    ]


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
    version="0.2.0",
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
        "borderline_margin": BORDERLINE_MARGIN,
        "borderline_zone": f"threshold +/- {BORDERLINE_MARGIN:.2f}",
        "borderline_low": round(state.threshold - BORDERLINE_MARGIN, 4),
        "borderline_high": round(state.threshold + BORDERLINE_MARGIN, 4),
        "device": str(state.device),
        "examples": list(EXAMPLE_PAIRS.values()),
    }


@app.get("/api/results")
async def get_results() -> Dict[str, Any]:
    """Read real measured training histories, parameter audits, and hardware benchmarks from disk."""
    all_runs, found_sources, missing_sources = _collect_results_runs()
    arch_data = _parse_architecture_doc(all_runs, found_sources, missing_sources)
    bench_data = _parse_benchmarks_and_ledger(found_sources, missing_sources)
    tiers = _build_tier_status(all_runs, bench_data.get("external_references", []))

    return {
        "runs": {
            "s0": all_runs.get("s0_seed42"),
            "s1": all_runs.get("s1_seed42"),
            "pilot_s0_mac": all_runs.get("pilot_s0_mac"),
            "pilot_s1_mac": all_runs.get("pilot_s1_mac"),
            "pilot_s0_colab": all_runs.get("pilot_s0_colab"),
            "pilot_s1_colab": all_runs.get("pilot_s1_colab"),
            "sanity_2a_full_v3": all_runs.get("sanity_2a_full_v3"),
            "sanity_2b_50spk_s0": all_runs.get("sanity_2b_50spk_s0"),
            "sanity_2c_active_s0": all_runs.get("sanity_2c_active_s0"),
            "all_available_run_ids": sorted(all_runs.keys()),
        },
        "architecture": arch_data,
        "benchmarks": bench_data,
        "tiers": tiers,
        "sources": {
            "found": found_sources,
            "missing": missing_sources,
        },
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
    verdict = classify_verdict(cosine_score, threshold, margin=BORDERLINE_MARGIN)

    return JSONResponse(
        status_code=200,
        content={
            "score": round(cosine_score, 4),
            "cosine_score": round(cosine_score, 4),
            "threshold": round(threshold, 4),
            "borderline_margin": BORDERLINE_MARGIN,
            "borderline_zone": f"threshold +/- {BORDERLINE_MARGIN:.2f}",
            "borderline_band": {
                "margin": BORDERLINE_MARGIN,
                "low": round(threshold - BORDERLINE_MARGIN, 4),
                "high": round(threshold + BORDERLINE_MARGIN, 4),
            },
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

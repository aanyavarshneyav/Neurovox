#!/usr/bin/env python3
"""Train-probe evaluation script for speaker verification models.

Evaluates trained model checkpoints on training-speaker verification trials
using the exact same evaluation metrics and feature extraction pipeline as
dev-clean and test-clean (full-utterance, no augmentation, fp32, FC1 L2-norm,
cosine scoring, threshold-free EER, minDCF, and speaker-level bootstrap CIs).

This enables like-for-like comparison between training verification performance
and generalization performance on dev/test sets.

Strictly read-only with respect to training artifacts: NEVER modifies history.csv,
checkpoints, or training configs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.trial_generator import TrialGenerator, save_trials_and_report
from src.eval.extractor import EmbeddingExtractor
from src.eval.metrics import compute_eer, compute_mindcf, speaker_bootstrap_ci
from src.models import (
    ECAPATDNN,
    build_ska_hnn,
    build_ska_hnn_from_config,
    ecapa_tdnn_standard,
    small_tdnn_se,
)
from src.utils.paths import get_data_root, get_results_root, resolve_path


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_config_hash(config_path: Path) -> str:
    """Compute 8-character hex digest of config file content."""
    content = config_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:8]


def load_config(config_path: Path) -> Any:
    """Load YAML config file."""
    from src.training.trainer import load_config as _load_cfg
    return _load_cfg(config_path)


def resolve_run_and_checkpoint(
    run_arg: str,
    ckpt_arg: str = "best_dev_eer.pt",
) -> Tuple[str, Path, Path]:
    """Resolve run directory, run ID, and checkpoint path.

    Args:
        run_arg: Run ID (e.g. 's0_seed42') or run directory path.
        ckpt_arg: Checkpoint filename ('best_dev_eer.pt', 'last.pt') or explicit path.

    Returns:
        Tuple of (run_id, run_dir, checkpoint_path).
    """
    results_root = get_results_root()
    repo_root = REPO_ROOT

    # Determine run_id and run_dir
    run_path = Path(run_arg).expanduser()
    if run_path.exists() and run_path.is_dir():
        run_dir = run_path.resolve()
        run_id = run_dir.name
    elif (results_root / run_arg).exists():
        run_dir = (results_root / run_arg).resolve()
        run_id = run_arg
    else:
        # Check if checkpoints/<run_arg> exists
        if (repo_root / "checkpoints" / run_arg).exists():
            run_dir = (results_root / run_arg).resolve()
            run_id = run_arg
        else:
            run_dir = (results_root / run_arg).resolve()
            run_id = Path(run_arg).name

    # Determine checkpoint path
    ckpt_path_candidate = Path(ckpt_arg).expanduser()
    if ckpt_path_candidate.is_absolute() and ckpt_path_candidate.exists():
        checkpoint_path = ckpt_path_candidate.resolve()
    elif ckpt_path_candidate.exists():
        checkpoint_path = ckpt_path_candidate.resolve()
    else:
        # Search candidate locations
        search_dirs = [
            repo_root / "checkpoints" / run_id,
            run_dir,
            repo_root / "checkpoints",
        ]
        found = None
        for d in search_dirs:
            p = d / ckpt_arg
            if p.exists():
                found = p.resolve()
                break
        if found is None:
            raise FileNotFoundError(
                f"Checkpoint '{ckpt_arg}' not found for run '{run_id}'. "
                f"Checked directories: {[str(d) for d in search_dirs]}"
            )
        checkpoint_path = found

    return run_id, run_dir, checkpoint_path


def instantiate_model_for_checkpoint(
    ckpt_payload: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, str]:
    """Instantiate model using the existing model factory matching checkpoint config."""
    config_hash = ckpt_payload.get("config_hash", "")
    matched_cfg = None
    configs_dir = REPO_ROOT / "configs"

    if config_hash and configs_dir.exists():
        for cfg_path in sorted(configs_dir.glob("*.yaml")):
            try:
                if compute_config_hash(cfg_path) == config_hash:
                    matched_cfg = load_config(cfg_path)
                    break
            except Exception:
                continue

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

    # Fallback inspection of state_dict
    state_dict = ckpt_payload.get("model_state_dict", ckpt_payload)
    if any(k.startswith(("in_conv.", "block1.")) for k in state_dict):
        model = small_tdnn_se(in_features=80, channels=128, emb_dim=192).to(device)
        return model, "small_tdnn_se"
    if any(k.startswith("ecapa.") for k in state_dict):
        model = ecapa_tdnn_standard(in_features=80, emb_dim=192).to(device)
        return model, "ecapa_tdnn"

    # Default to SKA-HNN S0
    model = build_ska_hnn(switch="S0").to(device)
    return model, "ska_hnn (S0)"


def ensure_train_probe_trials(
    trials_path: Optional[Path] = None,
    targets_per_speaker: int = 20,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """Ensure train-probe trial list exists under data/trials/; generate if missing.

    Returns:
        Tuple of (trials_path, trials_meta_path).
    """
    data_root = get_data_root()
    trials_dir = data_root / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    if trials_path is None:
        trials_path = trials_dir / "trials_train-probe.txt"
    else:
        trials_path = resolve_path(trials_path)

    trials_meta_path = trials_path.parent / f"{trials_path.stem}_meta.csv"

    if not trials_path.exists() or not trials_meta_path.exists():
        manifest_path = data_root / "manifests" / "train-clean-100.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest train-clean-100.csv not found at {manifest_path}")

        print(f"Generating deterministic train-probe trials (targets_per_speaker={targets_per_speaker}, seed={seed})...")
        generator = TrialGenerator(
            manifest_path=manifest_path,
            min_duration=2.0,
            seed=seed,
        )
        trials, report = generator.generate_trials(
            targets_per_speaker=targets_per_speaker,
            seed=seed,
        )
        out_report_path = trials_dir / "trials_report_train-probe.json"
        save_trials_and_report(trials, report, trials_path, out_report_path, trials_meta_path)

    return trials_path, trials_meta_path


def run_train_probe_evaluation(
    model: torch.nn.Module,
    trials_path: Path,
    trials_meta_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    bootstrap_iterations: int = 1000,
    seed: int = 42,
    max_trials: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Evaluate model on trial list with full-utterance extraction and bootstrap CIs.

    Args:
        model: Speaker embedding model in eval mode.
        trials_path: Path to trial list file (<label> <path_a> <path_b>).
        trials_meta_path: Optional companion metadata CSV.
        device: Device to use for extraction.
        bootstrap_iterations: Number of speaker-level bootstrap resamples.
        seed: Random seed for bootstrap reproducibility.
        max_trials: Optional maximum trial count to evaluate (for testing/partial runs).
        show_progress: Whether to show progress during extraction.

    Returns:
        Dictionary of evaluated metrics and trial statistics.
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    extractor = EmbeddingExtractor(model, device=device, batch_size=1)

    # Read trial file lines
    trial_lines: List[Tuple[int, str, str]] = []
    with open(trials_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                trial_lines.append((int(parts[0]), parts[1], parts[2]))

    if max_trials is not None and max_trials > 0 and max_trials < len(trial_lines):
        # Subset trials while maintaining balance if possible
        tgts = [t for t in trial_lines if t[0] == 1]
        imps = [t for t in trial_lines if t[0] == 0]
        half = max_trials // 2
        trial_lines = tgts[:half] + imps[:max_trials - half]

    # Collect unique audio paths participating in evaluated trials
    unique_paths = set()
    for _, pa, pb in trial_lines:
        unique_paths.add(pa)
        unique_paths.add(pb)

    t0 = time.perf_counter()
    embs: Dict[str, np.ndarray] = {}
    total_utts = len(unique_paths)
    for idx, path in enumerate(sorted(unique_paths), start=1):
        if show_progress and (idx == 1 or idx % 500 == 0 or idx == total_utts):
            print(f"Extracting embeddings: [{idx}/{total_utts}] ({idx/total_utts*100:.1f}%)")
        embs[path] = extractor.extract_from_file(path)
    extract_time = time.perf_counter() - t0

    # Score trials
    labels: List[int] = []
    scores: List[float] = []
    for lbl, pa, pb in trial_lines:
        ea = embs[pa]
        eb = embs[pb]
        # Unit-norm vectors: cosine similarity is simple dot product
        score = float(np.dot(ea, eb))
        labels.append(lbl)
        scores.append(score)

    labels_arr = np.array(labels, dtype=np.int32)
    scores_arr = np.array(scores, dtype=np.float32)

    # Point estimates
    eer, eer_thresh = compute_eer(scores_arr, labels_arr, strict=False)
    mindcf_05 = compute_mindcf(scores_arr, labels_arr, p_target=0.05, normalize=True, strict=False)
    mindcf_01 = compute_mindcf(scores_arr, labels_arr, p_target=0.01, normalize=True, strict=False)

    # Load speaker metadata for speaker-level clustered bootstrap CIs
    spk_a_list: List[str] = []
    spk_b_list: List[str] = []

    if trials_meta_path is not None and trials_meta_path.exists():
        df_meta = pd.read_csv(trials_meta_path)
        # Create lookup by (path_a, path_b)
        meta_lut: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for _, row in df_meta.iterrows():
            meta_lut[(str(row["path_a"]), str(row["path_b"]))] = (str(row["spk_a"]), str(row["spk_b"]))

        for _, pa, pb in trial_lines:
            if (pa, pb) in meta_lut:
                sa, sb = meta_lut[(pa, pb)]
            elif (pb, pa) in meta_lut:
                sb, sa = meta_lut[(pb, pa)]
            else:
                # Derive speaker ID from standard LibriSpeech path: <spk>/<chapter>/<spk>-<chapter>-<utt>.flac
                sa = Path(pa).name.split("-")[0]
                sb = Path(pb).name.split("-")[0]
            spk_a_list.append(sa)
            spk_b_list.append(sb)
    else:
        # Fallback to deriving from file path
        for _, pa, pb in trial_lines:
            sa = Path(pa).name.split("-")[0]
            sb = Path(pb).name.split("-")[0]
            spk_a_list.append(sa)
            spk_b_list.append(sb)

    unique_speakers = sorted(set(spk_a_list).union(set(spk_b_list)))

    bootstrap_ci: Optional[Dict[str, float]] = None
    if len(unique_speakers) >= 2 and bootstrap_iterations > 0 and len(labels_arr) >= 4:
        try:
            boot_res = speaker_bootstrap_ci(
                scores_a=scores_arr,
                labels=labels_arr,
                spk_a=spk_a_list,
                spk_b=spk_b_list,
                n_bootstrap=bootstrap_iterations,
                confidence_level=0.95,
                seed=seed,
            )
            ci_eer = boot_res["ci_95_a"]["eer"]
            ci_dcf05 = boot_res["ci_95_a"]["mindcf_p05"]
            ci_dcf01 = boot_res["ci_95_a"]["mindcf_p01"]
            bootstrap_ci = {
                "eer_low_pct": round(float(ci_eer[0]) * 100.0, 2),
                "eer_high_pct": round(float(ci_eer[1]) * 100.0, 2),
                "mindcf_p05_norm_low": round(float(ci_dcf05[0]), 4),
                "mindcf_p05_norm_high": round(float(ci_dcf05[1]), 4),
                "mindcf_p01_norm_low": round(float(ci_dcf01[0]), 4),
                "mindcf_p01_norm_high": round(float(ci_dcf01[1]), 4),
            }
        except Exception as e:
            print(f"Warning: speaker-level bootstrap CI computation skipped: {e}", file=sys.stderr)

    return {
        "trials_path": str(trials_path),
        "trials_sha256": compute_sha256(trials_path),
        "num_trials": len(labels_arr),
        "num_targets": int((labels_arr == 1).sum()),
        "num_impostors": int((labels_arr == 0).sum()),
        "num_speakers": len(unique_speakers),
        "extraction_time_sec": round(extract_time, 2),
        "eer": round(float(eer), 4),
        "eer_pct": round(float(eer) * 100.0, 2),
        "eer_threshold": round(float(eer_thresh), 4),
        "mindcf_p05": {
            "norm": round(float(mindcf_05.min_dcf), 4),
            "raw": round(float(mindcf_05.min_dcf_raw), 4),
            "threshold": round(float(mindcf_05.threshold), 4),
        },
        "mindcf_p01": {
            "norm": round(float(mindcf_01.min_dcf), 4),
            "raw": round(float(mindcf_01.min_dcf_raw), 4),
            "threshold": round(float(mindcf_01.threshold), 4),
        },
        "bootstrap_ci_95": bootstrap_ci,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate speaker verification model on training-speaker probe trials (read-only)."
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run identifier or run directory (e.g. 's0_seed42' or 'results/s0_seed42').",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="best_dev_eer.pt",
        help="Checkpoint filename ('best_dev_eer.pt', 'last.pt') or full path (default: 'best_dev_eer.pt').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Compute device for evaluation (default: 'cpu').",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: results/<run>/train_probe.json).",
    )
    parser.add_argument(
        "--trials",
        type=str,
        default=None,
        help="Path to train probe trial file (default: data/trials/trials_train-probe.txt).",
    )
    parser.add_argument(
        "--trials-meta",
        type=str,
        default=None,
        help="Path to companion metadata CSV (default: auto-detected beside trial file).",
    )
    parser.add_argument(
        "--targets-per-speaker",
        type=int,
        default=20,
        help="Number of target trials per qualifying speaker if generating trial list (default: 20).",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for 95% CIs (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic trial generation and bootstrap (default: 42).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Maximum number of trials to evaluate (useful for quick verification/testing).",
    )
    parser.add_argument(
        "--generate-trials-only",
        action="store_true",
        help="Only generate the trial list under data/trials/ and exit without running evaluation.",
    )
    args = parser.parse_args()

    # 1. Ensure trials exist
    trials_path, trials_meta_path = ensure_train_probe_trials(
        trials_path=Path(args.trials) if args.trials else None,
        targets_per_speaker=args.targets_per_speaker,
        seed=args.seed,
    )

    if args.generate_trials_only:
        print(f"Trials successfully ensured at: {trials_path}")
        print(f"SHA-256: {compute_sha256(trials_path)}")
        return 0

    # 2. Resolve run and checkpoint
    run_id, run_dir, ckpt_path = resolve_run_and_checkpoint(args.run, args.ckpt)
    out_path = Path(args.out).resolve() if args.out else (run_dir / "train_probe.json").resolve()

    device = torch.device(args.device)

    print("=" * 70)
    print("NEUROVOX TRAIN-PROBE EVALUATION")
    print(f"Run ID:          {run_id}")
    print(f"Checkpoint:      {ckpt_path}")
    print(f"Device:          {device}")
    print(f"Trials file:     {trials_path}")
    print(f"Output JSON:     {out_path}")
    print("=" * 70)

    # 3. Load checkpoint
    ckpt_payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = ckpt_payload.get("epoch", None)
    best_dev_eer = ckpt_payload.get("best_dev_eer", None)

    # 4. Instantiate model and load state
    model, arch_name = instantiate_model_for_checkpoint(ckpt_payload, device)
    state_dict = ckpt_payload.get("model_state_dict", ckpt_payload)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model architecture: {arch_name} (checkpoint epoch {epoch})")

    # 5. Run evaluation
    eval_res = run_train_probe_evaluation(
        model=model,
        trials_path=trials_path,
        trials_meta_path=trials_meta_path,
        device=device,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        max_trials=args.max_trials,
        show_progress=True,
    )

    # 6. Format final JSON output
    out_payload: Dict[str, Any] = {
        "run_id": run_id,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": compute_sha256(ckpt_path),
        "checkpoint_epoch": int(epoch) if epoch is not None else None,
        "best_dev_eer_stored": float(best_dev_eer) if best_dev_eer is not None else None,
        "model_architecture": arch_name,
        "num_speakers": eval_res["num_speakers"],
        "num_trials": eval_res["num_trials"],
        "num_targets": eval_res["num_targets"],
        "num_impostors": eval_res["num_impostors"],
        "trials_path": eval_res["trials_path"],
        "trials_sha256": eval_res["trials_sha256"],
        "extraction_time_sec": eval_res["extraction_time_sec"],
        "eer": eval_res["eer"],
        "eer_pct": eval_res["eer_pct"],
        "eer_threshold": eval_res["eer_threshold"],
        "mindcf_p05": eval_res["mindcf_p05"],
        "mindcf_p01": eval_res["mindcf_p01"],
        "bootstrap_ci_95": eval_res["bootstrap_ci_95"],
        "device": str(device),
        "precision": "fp32",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("TRAIN-PROBE RESULTS")
    print("=" * 60)
    print(f"Total Trials:      {eval_res['num_trials']:,} ({eval_res['num_targets']:,} tgt, {eval_res['num_impostors']:,} imp)")
    print(f"Unique Speakers:   {eval_res['num_speakers']}")
    print(f"Extraction Time:   {eval_res['extraction_time_sec']:.2f} s")
    print(f"Train-Probe EER:   {eval_res['eer_pct']:.2f}% (thresh: {eval_res['eer_threshold']:.4f})")
    print(f"minDCF (P=0.05):   norm={eval_res['mindcf_p05']['norm']:.4f} (raw={eval_res['mindcf_p05']['raw']:.4f})")
    print(f"minDCF (P=0.01):   norm={eval_res['mindcf_p01']['norm']:.4f} (raw={eval_res['mindcf_p01']['raw']:.4f})")
    if eval_res.get("bootstrap_ci_95"):
        ci = eval_res["bootstrap_ci_95"]
        print(f"95% CI EER:        [{ci['eer_low_pct']:.2f}%, {ci['eer_high_pct']:.2f}%]")
        print(f"95% CI minDCF_05:  [{ci['mindcf_p05_norm_low']:.4f}, {ci['mindcf_p05_norm_high']:.4f}]")
    print("=" * 60)

    # 7. Write to output file (and NEVER write to history.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved train-probe results to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

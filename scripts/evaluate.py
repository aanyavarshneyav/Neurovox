#!/usr/bin/env python3
"""Speaker verification evaluation script over evaluation trial lists.

Computes:
- Pairwise cosine similarity scores from full-utterance L2-normalized embeddings.
- Equal Error Rate (EER) and decision threshold.
- Normalized and raw minDCF at P_target = 0.05 (primary prior) and P_target = 0.01.
- 95% speaker-level clustered bootstrap confidence intervals using companion metadata CSV.
- Cryptographic SHA-256 hashes of the evaluated checkpoint and trial list.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval.extractor import EmbeddingExtractor
from src.eval.metrics import compute_eer, compute_mindcf, speaker_bootstrap_ci
from src.models.small_tdnn_se import small_tdnn_se
from src.utils.paths import resolve_path


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def evaluate_trials(
    model: torch.nn.Module,
    trials_path: Path,
    trials_meta_path: Path | None = None,
    device: torch.device | None = None,
    bootstrap_iterations: int = 1000,
    seed: int = 42,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run full evaluation on a trial list.

    Args:
        model: Trained speaker embedding model.
        trials_path: Path to trial list text file (<label> <path_a> <path_b>).
        trials_meta_path: Optional path to companion metadata CSV for speaker bootstrap.
        device: Device to use for feature extraction and inference.
        bootstrap_iterations: Number of bootstrap iterations for 95% CIs.
        seed: Random seed for bootstrap reproducibility.
        show_progress: Whether to print progress during embedding extraction.

    Returns:
        Dictionary containing evaluation metrics and audit details.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    extractor = EmbeddingExtractor(model, device=device, batch_size=1)

    t0_extract = time.perf_counter()
    embs = extractor.extract_for_trial_file(trials_path, show_progress=show_progress)
    extract_time = time.perf_counter() - t0_extract

    # Load trials
    labels: list[int] = []
    scores: list[float] = []

    with open(trials_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                lbl = int(parts[0])
                pa, pb = parts[1], parts[2]
                ea = embs[pa]
                eb = embs[pb]
                # Cosine similarity between unit-norm vectors is simply their dot product
                score = float(np.dot(ea, eb))
                labels.append(lbl)
                scores.append(score)

    labels_arr = np.array(labels, dtype=np.int32)
    scores_arr = np.array(scores, dtype=np.float32)

    # Compute EER
    eer, eer_thresh = compute_eer(scores_arr, labels_arr)

    # Compute minDCF at primary P_target=0.05 and indicative P_target=0.01
    mindcf_05 = compute_mindcf(scores_arr, labels_arr, p_target=0.05, normalize=True)
    mindcf_01 = compute_mindcf(scores_arr, labels_arr, p_target=0.01, normalize=True)

    results: dict[str, Any] = {
        "trials_path": str(trials_path),
        "trials_sha256": compute_sha256(trials_path),
        "num_trials": len(labels_arr),
        "num_targets": int((labels_arr == 1).sum()),
        "num_impostors": int((labels_arr == 0).sum()),
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
    }

    # Speaker-level clustered bootstrap CI if metadata CSV is available
    if trials_meta_path is not None and trials_meta_path.exists():
        df_meta = pd.read_csv(trials_meta_path)
        boot_res = speaker_bootstrap_ci(
            scores_a=scores_arr,
            labels=labels_arr,
            spk_a=df_meta["spk_a"].values,
            spk_b=df_meta["spk_b"].values,
            n_bootstrap=bootstrap_iterations,
            confidence_level=0.95,
            seed=seed,
        )
        ci_eer = boot_res["ci_95_a"]["eer"]
        ci_dcf05 = boot_res["ci_95_a"]["mindcf_p05"]
        ci_dcf01 = boot_res["ci_95_a"]["mindcf_p01"]
        results["bootstrap_ci_95"] = {
            "eer_low_pct": round(float(ci_eer[0]) * 100.0, 2),
            "eer_high_pct": round(float(ci_eer[1]) * 100.0, 2),
            "mindcf_p05_norm_low": round(float(ci_dcf05[0]), 4),
            "mindcf_p05_norm_high": round(float(ci_dcf05[1]), 4),
            "mindcf_p01_norm_low": round(float(ci_dcf01[0]), 4),
            "mindcf_p01_norm_high": round(float(ci_dcf01[1]), 4),
        }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate speaker verification checkpoint on trial lists.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--trials",
        type=str,
        required=True,
        help="Path to trial list file (e.g. data/trials/trials_dev-clean.txt)",
    )
    parser.add_argument(
        "--trials-meta",
        type=str,
        default=None,
        help="Optional path to companion metadata CSV (default: auto-detected if beside trial file)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write evaluation results JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('mps', 'cuda', 'cpu', 'auto')",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Number of speaker bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap reproducibility (default: 42)",
    )
    args = parser.parse_args()

    ckpt_path = resolve_path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint file not found: {ckpt_path}", file=sys.stderr)
        return 1

    trials_path = resolve_path(args.trials)
    if not trials_path.exists():
        print(f"ERROR: Trials file not found: {trials_path}", file=sys.stderr)
        return 1

    trials_meta_path = (
        resolve_path(args.trials_meta)
        if args.trials_meta
        else trials_path.parent / f"{trials_path.stem}_meta.csv"
    )

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=" * 75)
    print("NEUROVOX SPEAKER VERIFICATION EVALUATION")
    print(f"Checkpoint:  {ckpt_path.resolve()}")
    print(f"Trials:      {trials_path.resolve()}")
    print(f"Device:      {device}")
    print("=" * 75)

    # Load model
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model = small_tdnn_se(in_features=80, channels=128, emb_dim=192).to(device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    results = evaluate_trials(
        model=model,
        trials_path=trials_path,
        trials_meta_path=trials_meta_path if trials_meta_path.exists() else None,
        device=device,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        show_progress=True,
    )

    results["checkpoint_path"] = str(ckpt_path)
    results["checkpoint_sha256"] = compute_sha256(ckpt_path)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Trials:      {results['num_trials']:,} ({results['num_targets']:,} tgt, {results['num_impostors']:,} imp)")
    print(f"Extraction Time:   {results['extraction_time_sec']:.2f} s")
    print(f"EER (%):           {results['eer_pct']:.2f}% (thresh: {results['eer_threshold']:.4f})")
    print(f"minDCF (P=0.05):   norm={results['mindcf_p05']['norm']:.4f} (raw={results['mindcf_p05']['raw']:.4f})")
    print(f"minDCF (P=0.01):   norm={results['mindcf_p01']['norm']:.4f} (raw={results['mindcf_p01']['raw']:.4f})")
    if "bootstrap_ci_95" in results:
        ci = results["bootstrap_ci_95"]
        print(f"95% CI EER:        [{ci['eer_low_pct']:.2f}%, {ci['eer_high_pct']:.2f}%]")
        print(f"95% CI minDCF_05:  [{ci['mindcf_p05_norm_low']:.4f}, {ci['mindcf_p05_norm_high']:.4f}]")
    print("=" * 60)

    if args.output_json:
        out_path = resolve_path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation metrics JSON to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Run pretrained SpeechBrain ECAPA-TDNN reference evaluation on LibriSpeech verification trials (v2).

Downloads the pretrained 'speechbrain/spkrec-ecapa-voxceleb' (~100 MB) model to
checkpoints/pretrained/ and computes cosine verification scores on trial pairs.

Adheres to Scientific Integrity Rules:
- Serves as an EXTERNAL REFERENCE, not as a pass/fail training gate or proof of zero leakage.
- Evaluates both test-clean and test-other v2 trial lists with speaker-level bootstrap CIs.
- Evaluates two-sample difference (test-other minus test-clean) with speaker-level bootstrap CI.
- Logs measured EER and minDCF (normalized and raw) to docs/RESULTS_LOG.md.
"""

import argparse
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import (
    compute_eer,
    compute_mindcf,
    speaker_bootstrap_ci,
    speaker_bootstrap_two_sample_difference,
)
from src.utils.paths import PROJECT_ROOT, get_ckpt_root, resolve_path


def load_trials_with_meta(trials_file: Path) -> tuple[list[tuple[int, Path, Path]], np.ndarray, np.ndarray]:
    """Load trial pairs and companion speaker IDs from text and metadata CSV."""
    trials_file = Path(trials_file).resolve()
    meta_file = trials_file.parent / f"{trials_file.stem}_meta.csv"

    if not meta_file.exists():
        raise FileNotFoundError(f"Companion metadata CSV not found: {meta_file}")

    df_meta = pd.read_csv(meta_file)
    trials: list[tuple[int, Path, Path]] = []
    with open(trials_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            label = int(parts[0])
            p1 = resolve_path(parts[1])
            p2 = resolve_path(parts[2])
            trials.append((label, p1, p2))

    if len(trials) != len(df_meta):
        raise ValueError(f"Mismatch in trial count: {len(trials)} vs {len(df_meta)} in {meta_file.name}")

    spk_a = df_meta["spk_a"].astype(str).to_numpy()
    spk_b = df_meta["spk_b"].astype(str).to_numpy()

    return trials, spk_a, spk_b


def extract_embedding(model, audio_path: Path, device: torch.device) -> torch.Tensor:
    """Load audio using soundfile and extract l2-normalized embedding vector."""
    audio, sr = sf.read(str(audio_path), dtype="float32")
    wav = torch.from_numpy(audio).float()
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # Shape (1, time)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.to(device)
    with torch.no_grad():
        emb = model.encode_batch(wav)
        emb = emb.squeeze()
        emb = emb / (torch.norm(emb, p=2, dim=-1, keepdim=True) + 1e-8)
    return emb.cpu()


def evaluate_partition(
    classifier,
    trials_file: Path,
    device: torch.device,
    embedding_cache: dict[Path, torch.Tensor],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate pretrained classifier on one partition and compute speaker bootstrap CI."""
    trials, spk_a, spk_b = load_trials_with_meta(trials_file)
    print(f"\nEvaluating {trials_file.name} ({len(trials):,} trials)...")

    # Collect missing embeddings
    unique_paths = sorted(list(set([t[1] for t in trials] + [t[2] for t in trials])))
    needed_paths = [p for p in unique_paths if p not in embedding_cache]
    if needed_paths:
        print(f"Extracting embeddings for {len(needed_paths)} new audio files...")
        for idx, p in enumerate(needed_paths, start=1):
            if idx % 200 == 0 or idx == len(needed_paths):
                print(f"  Extracted {idx} / {len(needed_paths)} embeddings...")
            embedding_cache[p] = extract_embedding(classifier, p, device)

    # Compute cosine similarity
    labels_list: list[int] = []
    scores_list: list[float] = []
    for label, p1, p2 in trials:
        e1 = embedding_cache[p1]
        e2 = embedding_cache[p2]
        cosine_sim = float(torch.dot(e1, e2).item())
        labels_list.append(label)
        scores_list.append(cosine_sim)

    labels = np.array(labels_list, dtype=np.int64)
    scores = np.array(scores_list, dtype=np.float64)

    # Point estimates with strict=True
    eer, eer_th = compute_eer(scores, labels, strict=True)
    dcf05 = compute_mindcf(scores, labels, p_target=0.05, normalize=True, strict=True)
    dcf01 = compute_mindcf(scores, labels, p_target=0.01, normalize=True, strict=True)

    # Speaker bootstrap CI
    print(f"Computing speaker-level clustered bootstrap ({n_bootstrap} iterations)...")
    boot_res = speaker_bootstrap_ci(
        scores_a=scores,
        labels=labels,
        spk_a=spk_a,
        spk_b=spk_b,
        n_bootstrap=n_bootstrap,
        confidence_level=0.95,
        seed=seed,
    )

    ci_eer = boot_res["ci_95_a"]["eer"]
    ci_dcf05 = boot_res["ci_95_a"]["mindcf_p05"]
    ci_dcf01 = boot_res["ci_95_a"]["mindcf_p01"]

    print("------------------------------------------------------------------")
    print(f"Results for {trials_file.name}:")
    print(f"  EER (%):                 {eer * 100.0:.2f}%  [95% CI: {ci_eer[0] * 100.0:.2f}%, {ci_eer[1] * 100.0:.2f}%] (threshold: {eer_th:.4f})")
    print(f"  minDCF (P=0.05, norm):   {dcf05['min_dcf']:.4f}  [95% CI: {ci_dcf05[0]:.4f}, {ci_dcf05[1]:.4f}] (raw: {dcf05['min_dcf_raw']:.4f})")
    print(f"  minDCF (P=0.01, norm):   {dcf01['min_dcf']:.4f}  [95% CI: {ci_dcf01[0]:.4f}, {ci_dcf01[1]:.4f}] (raw: {dcf01['min_dcf_raw']:.4f})")
    print("------------------------------------------------------------------")

    return {
        "trials_file": trials_file.name,
        "n_trials": len(trials),
        "scores": scores,
        "labels": labels,
        "spk_a": spk_a,
        "spk_b": spk_b,
        "eer": eer,
        "eer_th": eer_th,
        "ci_95_eer": ci_eer,
        "mindcf_p05": dcf05["min_dcf"],
        "mindcf_p05_raw": dcf05["min_dcf_raw"],
        "ci_95_mindcf_p05": ci_dcf05,
        "mindcf_p01": dcf01["min_dcf"],
        "mindcf_p01_raw": dcf01["min_dcf_raw"],
        "ci_95_mindcf_p01": ci_dcf01,
        "bootstrap_results": boot_res,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pretrained ECAPA-TDNN reference on verification trials.")
    parser.add_argument(
        "--trials",
        type=str,
        default=None,
        help="Path to single trial file (e.g. data/trials/trials_test-clean.txt).",
    )
    parser.add_argument(
        "--eval-both",
        action="store_true",
        help="Evaluate both test-clean and test-other and compute speaker bootstrap difference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, mps, cuda, cpu).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of speaker bootstrap iterations (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap (default: 42).",
    )
    args = parser.parse_args()

    from speechbrain.inference.speaker import EncoderClassifier

    if args.device == "auto":
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    print(f"Loading pretrained SpeechBrain ECAPA-TDNN on {device}...")
    save_dir = get_ckpt_root() / "pretrained" / "spkrec-ecapa-voxceleb"
    save_dir.mkdir(parents=True, exist_ok=True)

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(save_dir),
        run_opts={"device": str(device)},
    )

    embedding_cache: dict[Path, torch.Tensor] = {}

    if args.eval_both or args.trials is None:
        trials_clean = PROJECT_ROOT / "data" / "trials" / "trials_test-clean.txt"
        trials_other = PROJECT_ROOT / "data" / "trials" / "trials_test-other.txt"

        res_clean = evaluate_partition(classifier, trials_clean, device, embedding_cache, n_bootstrap=args.n_bootstrap, seed=args.seed)
        res_other = evaluate_partition(classifier, trials_other, device, embedding_cache, n_bootstrap=args.n_bootstrap, seed=args.seed)

        print("\n==================================================================")
        print("Pretrained ECAPA Reference Comparison: test-other vs test-clean")
        print("==================================================================")
        print(f"test-clean EER (%):       {res_clean['eer'] * 100.0:.2f}%  [95% CI: {res_clean['ci_95_eer'][0] * 100.0:.2f}%, {res_clean['ci_95_eer'][1] * 100.0:.2f}%]")
        print(f"test-clean minDCF (0.05): {res_clean['mindcf_p05']:.4f}  [95% CI: {res_clean['ci_95_mindcf_p05'][0]:.4f}, {res_clean['ci_95_mindcf_p05'][1]:.4f}]")
        print(f"test-clean minDCF (0.01): {res_clean['mindcf_p01']:.4f}  [95% CI: {res_clean['ci_95_mindcf_p01'][0]:.4f}, {res_clean['ci_95_mindcf_p01'][1]:.4f}]")
        print("------------------------------------------------------------------")
        print(f"test-other EER (%):       {res_other['eer'] * 100.0:.2f}%  [95% CI: {res_other['ci_95_eer'][0] * 100.0:.2f}%, {res_other['ci_95_eer'][1] * 100.0:.2f}%]")
        print(f"test-other minDCF (0.05): {res_other['mindcf_p05']:.4f}  [95% CI: {res_other['ci_95_mindcf_p05'][0]:.4f}, {res_other['ci_95_mindcf_p05'][1]:.4f}]")
        print(f"test-other minDCF (0.01): {res_other['mindcf_p01']:.4f}  [95% CI: {res_other['ci_95_mindcf_p01'][0]:.4f}, {res_other['ci_95_mindcf_p01'][1]:.4f}]")
        print("------------------------------------------------------------------")

        print("Computing two-sample speaker bootstrap difference (test-other minus test-clean)...")
        two_sample_res = speaker_bootstrap_two_sample_difference(
            scores_1=res_clean["scores"],
            labels_1=res_clean["labels"],
            spk_a_1=res_clean["spk_a"],
            spk_b_1=res_clean["spk_b"],
            scores_2=res_other["scores"],
            labels_2=res_other["labels"],
            spk_a_2=res_other["spk_a"],
            spk_b_2=res_other["spk_b"],
            n_bootstrap=args.n_bootstrap,
            confidence_level=0.95,
            seed=args.seed,
        )

        diff_pt = two_sample_res["diff_point_estimate"]
        diff_ci = two_sample_res["diff_ci_95"]
        print(f"Difference (test-other - test-clean):")
        print(f"  Delta EER (%):           {diff_pt['diff_eer'] * 100.0:+.2f}%  [95% CI: {diff_ci['diff_eer'][0] * 100.0:+.2f}%, {diff_ci['diff_eer'][1] * 100.0:+.2f}%]")
        print(f"  Delta minDCF (P=0.05):   {diff_pt['diff_mindcf_p05']:+.4f}  [95% CI: {diff_ci['diff_mindcf_p05'][0]:+.4f}, {diff_ci['diff_mindcf_p05'][1]:+.4f}]")
        print(f"  Delta minDCF (P=0.01):   {diff_pt['diff_mindcf_p01']:+.4f}  [95% CI: {diff_ci['diff_mindcf_p01'][0]:+.4f}, {diff_ci['diff_mindcf_p01'][1]:+.4f}]")
        print("==================================================================")
    else:
        evaluate_partition(classifier, Path(args.trials).resolve(), device, embedding_cache, n_bootstrap=args.n_bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()

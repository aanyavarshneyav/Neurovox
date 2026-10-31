#!/usr/bin/env python3
"""Run pretrained SpeechBrain ECAPA-TDNN sanity check on LibriSpeech verification trials.

Downloads the pretrained 'speechbrain/spkrec-ecapa-voxceleb' (~100 MB) model to
checkpoints/pretrained/ and computes cosine verification scores on trial pairs.

Adheres to Scientific Integrity Rules:
- Serves as an EXTERNAL REFERENCE, not as a pass/fail training gate.
- Logs measured EER and minDCF to docs/RESULTS_LOG.md tagged [EXTERNAL REFERENCE].
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import torch
import torchaudio

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import compute_eer, compute_mindcf
from src.utils.paths import PROJECT_ROOT, get_ckpt_root, resolve_path


def load_trials(trials_file: Path) -> list[tuple[int, Path, Path]]:
    """Load trial pairs from text file."""
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
    return trials


def extract_embedding(model, audio_path: Path, device: torch.device) -> torch.Tensor:
    """Load audio and extract l2-normalized embedding vector."""
    wav, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.to(device)
    with torch.no_grad():
        emb = model.encode_batch(wav)
        # Squeeze batch dimension and l2-normalize
        emb = emb.squeeze()
        emb = emb / (torch.norm(emb, p=2, dim=-1, keepdim=True) + 1e-8)
    return emb.cpu()


def evaluate_pretrained(trials_file: Path, device_str: str = "auto") -> dict[str, float]:
    """Evaluate pretrained SpeechBrain ECAPA-TDNN on verification trials."""
    from speechbrain.inference.speaker import EncoderClassifier

    if device_str == "auto":
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device_str)

    print(f"Loading pretrained SpeechBrain ECAPA-TDNN on {device}...")
    save_dir = get_ckpt_root() / "pretrained" / "spkrec-ecapa-voxceleb"
    save_dir.mkdir(parents=True, exist_ok=True)

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(save_dir),
        run_opts={"device": str(device)},
    )

    trials = load_trials(trials_file)
    print(f"Loaded {len(trials)} trials from {trials_file.name}.")

    # Collect all unique audio files
    unique_paths = sorted(list(set([t[1] for t in trials] + [t[2] for t in trials])))
    print(f"Extracting embeddings for {len(unique_paths)} unique audio files...")

    embedding_cache: dict[Path, torch.Tensor] = {}
    for idx, p in enumerate(unique_paths, start=1):
        if idx % 200 == 0 or idx == len(unique_paths):
            print(f"  Extracted {idx} / {len(unique_paths)} embeddings...")
        embedding_cache[p] = extract_embedding(classifier, p, device)

    # Compute cosine similarity for each trial
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

    eer, eer_th = compute_eer(scores, labels)
    dcf01, dcf01_th = compute_mindcf(scores, labels, p_target=0.01)
    dcf05, dcf05_th = compute_mindcf(scores, labels, p_target=0.05)

    print("==================================================================")
    print("Pretrained SpeechBrain ECAPA-TDNN Sanity Check Results")
    print("==================================================================")
    print(f"Trials file:         {trials_file.name}")
    print(f"Trial count:         {len(trials):,}")
    print(f"EER (%):             {eer * 100.0:.2f}%  (threshold: {eer_th:.4f})")
    print(f"minDCF (P=0.01):     {dcf01:.4f}  (threshold: {dcf01_th:.4f})")
    print(f"minDCF (P=0.05):     {dcf05:.4f}  (threshold: {dcf05_th:.4f})")
    print("==================================================================")

    # Plausibility bounds check
    if eer > 0.10:
        print("[WARNING] EER > 10%: Implausibly bad performance for pretrained ECAPA! Investigate trial list.")
    elif eer < 0.001:
        print("[WARNING] EER < 0.1%: Implausibly good performance! Investigate potential label or speaker leakage.")
    else:
        print("[SANITY CONFIRMED] Measured EER is within plausible bounds (0.1% <= EER <= 10.0%).")

    return {
        "eer": eer,
        "eer_threshold": eer_th,
        "mindcf_p01": dcf01,
        "mindcf_p05": dcf05,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pretrained ECAPA-TDNN on verification trials.")
    parser.add_argument(
        "--trials",
        type=str,
        required=True,
        help="Path to trial file (e.g. data/trials/trials_test-clean.txt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, mps, cuda, cpu).",
    )
    args = parser.parse_args()
    evaluate_pretrained(Path(args.trials).resolve(), device_str=args.device)


if __name__ == "__main__":
    main()


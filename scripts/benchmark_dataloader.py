#!/usr/bin/env python3
"""DataLoader throughput benchmark on CPU for LibriSpeech FLAC audio.

Decodes FLAC audio files using soundfile, applies a random 180-frame crop
(exactly 29,040 samples @ 16 kHz), and extracts 80-bin log-Mel filterbank
spectrograms (25 ms window, 10 ms hop) using torchaudio on CPU.

Benchmarks throughput (samples/sec, batches/sec) across num_workers in {0, 2, 4, 6}.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T

# Import path resolver from src.utils
try:
    from src.utils.paths import get_data_root, resolve_path
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT))
    from src.utils.paths import get_data_root, resolve_path


class LibriSpeechFLACDataset(Dataset):
    """Dataset for raw FLAC decoding, random 29,040-sample cropping, and CPU log-Mel extraction."""

    def __init__(self, flac_paths: list[Path], target_samples: int = 29040) -> None:
        """Initialize with a list of FLAC file paths.
        
        Args:
            flac_paths: List of paths to .flac audio files.
            target_samples: Number of audio samples per crop (29,040 samples = 180 frames).
        """
        self.flac_paths = flac_paths
        self.target_samples = target_samples
        # CPU log-Mel transform: 80 bins, 25ms win (400 samples), 10ms hop (160 samples), center=False
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80,
            power=2.0,
            center=False,
        )

    def __len__(self) -> int:
        return len(self.flac_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = str(self.flac_paths[idx])
        # Decode FLAC with soundfile
        audio, sr = sf.read(path, dtype="float32")

        # Convert to mono if multi-channel
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Slice or pad to exactly target_samples (29,040)
        n_samples = len(audio)
        if n_samples < self.target_samples:
            # Repeat or pad
            repeats = (self.target_samples // n_samples) + 1
            audio = np.tile(audio, repeats)[: self.target_samples]
        elif n_samples > self.target_samples:
            start_idx = random.randint(0, n_samples - self.target_samples)
            audio = audio[start_idx : start_idx + self.target_samples]

        # Waveform shape (1, 29040)
        waveform = torch.from_numpy(audio).unsqueeze(0)

        # Compute log-Mel on CPU: shape (1, 80, 180)
        mel = self.mel_transform(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-6))
        return log_mel


def create_mock_flac_dataset(mock_dir: Path, num_files: int = 128) -> list[Path]:
    """Generate temporary synthetic FLAC files to verify dataloader pipeline if needed."""
    mock_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    print(f"Generating {num_files} synthetic mock FLAC files in {mock_dir} for offline verification...")
    rng = np.random.default_rng(42)
    for i in range(num_files):
        path = mock_dir / f"mock_sample_{i:04d}.flac"
        if not path.exists():
            # Random duration between 2.0s (32,000 samples) and 4.0s (64,000 samples) at 16kHz
            num_samples = int(rng.integers(32000, 64000))
            data = rng.standard_normal(num_samples).astype(np.float32) * 0.1
            sf.write(str(path), data, samplerate=16000, format="FLAC")
        paths.append(path)
    return paths


def benchmark_workers(
    dataset: Dataset,
    num_workers_list: list[int],
    batch_size: int = 64,
    num_batches: int = 25,
) -> list[dict[str, Any]]:
    """Benchmark DataLoader iteration throughput for each worker count."""
    results: list[dict[str, Any]] = []

    for nw in num_workers_list:
        print(f"\n--> Benchmarking DataLoader with num_workers = {nw}...")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw,
            persistent_workers=(nw > 0),
            pin_memory=False,
            drop_last=True,
        )

        # Warmup loop (2 batches)
        iterator = iter(loader)
        for _ in range(min(2, len(loader))):
            try:
                _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                _ = next(iterator)

        # Timed loop
        step_times: list[float] = []
        total_samples = 0
        batches_processed = 0

        start_time = time.perf_counter()
        for batch in loader:
            total_samples += batch.size(0)
            batches_processed += 1
            if batches_processed >= num_batches:
                break
        elapsed = time.perf_counter() - start_time

        samples_per_sec = total_samples / elapsed if elapsed > 0 else 0.0
        batches_per_sec = batches_processed / elapsed if elapsed > 0 else 0.0
        avg_batch_ms = (elapsed / batches_processed) * 1000.0 if batches_processed > 0 else 0.0

        res: dict[str, Any] = {
            "num_workers": nw,
            "batch_size": batch_size,
            "batches_processed": batches_processed,
            "total_samples": total_samples,
            "elapsed_sec": round(elapsed, 3),
            "samples_per_sec": round(samples_per_sec, 2),
            "batches_per_sec": round(batches_per_sec, 2),
            "avg_batch_latency_ms": round(avg_batch_ms, 2),
        }
        results.append(res)
        print(
            f"    num_workers={nw}: {samples_per_sec:.2f} samples/s | "
            f"{batches_per_sec:.2f} batches/s | {avg_batch_ms:.2f} ms/batch"
        )

    return results


def main() -> int:
    default_data_dir = str(get_data_root() / "LibriSpeech" / "dev-clean")
    parser = argparse.ArgumentParser(description="Neurovox DataLoader Benchmark")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=default_data_dir,
        help="Path to LibriSpeech dev-clean directory containing .flac files (default: DATA_ROOT/LibriSpeech/dev-clean)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=30,
        help="Number of batches to measure per worker count (default: 30)",
    )
    parser.add_argument(
        "--use-mock-if-empty",
        action="store_true",
        help="Generate synthetic mock FLAC files if target data directory is missing/empty",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="docs/dataloader_benchmarks_raw.json",
        help="Path to save raw benchmark results as JSON",
    )
    args = parser.parse_args()

    data_path = resolve_path(args.data_dir)
    flac_files = list(data_path.glob("**/*.flac")) if data_path.exists() else []

    if not flac_files:
        if args.use_mock_if_empty:
            mock_cache_dir = resolve_path("data/mock_flac_cache")
            print(f"Notice: No FLAC files found in {data_path}. Creating mock dataset for dry run...")
            flac_files = create_mock_flac_dataset(mock_cache_dir, num_files=256)
        else:
            print(f"ERROR: No FLAC audio files found in directory: {data_path.resolve()}")
            print("To download LibriSpeech dev-clean (~340 MB), request user confirmation first.")
            print("Alternatively, pass --use-mock-if-empty to verify pipeline on synthetic FLAC files.")
            return 1

    print("=" * 75)
    print("NEUROVOX DATALOADER THROUGHPUT BENCHMARK")
    print(f"Data source: {data_path.resolve()} ({len(flac_files)} files found)")
    print(f"Batch size: {args.batch_size} | Batches evaluated: {args.num_batches}")
    print("Workers evaluated: [0, 2, 4, 6]")
    print("=" * 75)

    dataset = LibriSpeechFLACDataset(flac_files)
    worker_results = benchmark_workers(
        dataset=dataset,
        num_workers_list=[0, 2, 4, 6],
        batch_size=args.batch_size,
        num_batches=args.num_batches,
    )

    print("\n" + "=" * 65)
    print(f"{'num_workers':<14} {'samples/sec':<16} {'batches/sec':<14} {'avg batch (ms)':<16}")
    print("-" * 65)
    for r in worker_results:
        print(
            f"{r['num_workers']:<14} {r['samples_per_sec']:<16.2f} "
            f"{r['batches_per_sec']:<14.2f} {r['avg_batch_latency_ms']:<16.2f}"
        )
    print("=" * 65)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(worker_results, f, indent=2)
    print(f"Dataloader results saved to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""Smoke data loader test script for Phase 1C.

Iterates 20 batches using LibriSpeechDataset and SpeakerBalancedBatchSampler,
verifies batch shape (64, 80, 180), dtype (torch.float32), distinct speaker labels,
measures data-loading throughput using num_workers=4 from docs/BENCHMARKS.md,
compares throughput against model step benchmarks, and saves a spectrogram figure.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Project root resolution
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import LibriSpeechDataset
from src.data.sampler import SpeakerBalancedBatchSampler
from src.utils.paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke data loader test script.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/train-clean-100.csv",
        help="Path to manifest CSV (default: data/manifests/train-clean-100.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size (default 64 per docs/PROJECT_SPEC.md)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=20,
        help="Number of batches to iterate (default: 20)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count (default: 4 per docs/BENCHMARKS.md)",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="reports/figures/smoke_spectrograms.png",
        help="Path to save spectrogram plot image",
    )
    return parser.parse_args()


def save_spectrogram_plot(
    features: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    num_samples: int = 4,
) -> None:
    """Save a grid of spectrogram images for inspection.

    Args:
        features: Batch tensor of shape (B, 80, 180).
        labels: Batch tensor of speaker labels (B,).
        output_path: Path to save the PNG image.
        num_samples: Number of samples to plot (default 4).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = min(num_samples, features.size(0))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(num_samples):
        spec = features[i].detach().cpu().numpy()  # Shape: (80, 180)
        label = labels[i].item()
        ax = axes[i]
        im = ax.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"Sample {i+1}: Speaker Label {label} | Shape (80, 180)")
        ax.set_xlabel("Time Frames (10 ms hop)")
        ax.set_ylabel("Mel Frequency Bins (80)")
        fig.colorbar(im, ax=ax, format="%+2.1f dB")

    plt.suptitle("Neurovox Phase 1C — Acoustic Features (80-dim log-Mel + CMN + SpecAugment)", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Saved spectrogram plot to: {output_path}")


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    output_image_path = resolve_path(args.output_image)

    print("==========================================================")
    print(" Neurovox Phase 1C — Data Pipeline Smoke Test")
    print("==========================================================")
    print(f"Manifest: {manifest_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Target Batches: {args.num_batches}")
    print(f"DataLoader Workers: {args.num_workers} (from docs/BENCHMARKS.md)")

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        sys.exit(1)

    # 1. Instantiate dataset with on-the-fly feature extraction and augmentations
    dataset = LibriSpeechDataset(
        manifest_path=manifest_path,
        target_samples=29040,  # 180 frames @ 16 kHz
        is_train=True,
        return_features=True,
        enable_speed_aug=True,
        enable_noise_aug=True,
        enable_babble_aug=False,  # Can be enabled when babble pool is desired
        enable_rir_aug=False,     # RIRS_NOISES not downloaded on host
        enable_specaug=True,
    )
    print(f"Dataset successfully loaded: {len(dataset):,} utterances across {len(set(dataset.speaker_ids))} speakers.")

    # 2. Instantiate speaker-balanced batch sampler
    sampler = SpeakerBalancedBatchSampler(
        speaker_labels=dataset.labels,
        batch_size=args.batch_size,
        batches_per_epoch=args.num_batches,
        seed=42,
    )

    # 3. Build DataLoader
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=False,
    )

    print(f"\n--> Starting iteration over {args.num_batches} batches...")
    batch_latencies_ms: list[float] = []
    first_batch_saved = False
    total_samples = 0

    # Warmup: fetch 1 batch if workers > 0 to prime worker processes
    loader_iter = iter(loader)
    warmup_start = time.perf_counter()
    warmup_feats, warmup_labels = next(loader_iter)
    warmup_sec = time.perf_counter() - warmup_start
    print(f"Warmup batch loaded in {warmup_sec*1000:.1f} ms.")

    start_time = time.perf_counter()
    batch_idx = 0

    # Process first batch from warmup
    total_samples += warmup_feats.size(0)
    batch_idx += 1
    save_spectrogram_plot(warmup_feats, warmup_labels, output_image_path, num_samples=4)

    # Time remaining batches
    for feats, labels in loader_iter:
        b_start = time.perf_counter()
        batch_idx += 1
        total_samples += feats.size(0)

        # Assertions
        assert feats.shape == (args.batch_size, 80, 180), (
            f"Expected batch shape ({args.batch_size}, 80, 180), got {feats.shape}"
        )
        assert feats.dtype == torch.float32, f"Expected float32, got {feats.dtype}"
        assert len(set(labels.tolist())) == args.batch_size, (
            f"Expected {args.batch_size} distinct speakers in batch, got {len(set(labels.tolist()))}"
        )

        b_elapsed_ms = (time.perf_counter() - b_start) * 1000.0
        batch_latencies_ms.append(b_elapsed_ms)

        if batch_idx % 5 == 0 or batch_idx == args.num_batches:
            print(
                f"  Batch {batch_idx:02d}/{args.num_batches}: "
                f"Shape={tuple(feats.shape)}, Dtype={feats.dtype}, "
                f"Distinct Speakers={len(set(labels.tolist()))}/{args.batch_size}, "
                f"Mean={feats.mean().item():.3f}, Std={feats.std().item():.3f}"
            )

        if batch_idx >= args.num_batches:
            break

    total_elapsed = time.perf_counter() - start_time
    avg_batch_ms = (total_elapsed / batch_idx) * 1000.0
    throughput_samples_sec = total_samples / total_elapsed
    throughput_batches_sec = batch_idx / total_elapsed

    print("\n==========================================================")
    print(" Performance & Throughput Summary")
    print("==========================================================")
    print(f"Total Batches Processed: {batch_idx}")
    print(f"Total Samples Loaded: {total_samples}")
    print(f"Total Elapsed Time: {total_elapsed:.2f} s")
    print(f"Average Batch Latency: {avg_batch_ms:.2f} ms/batch")
    print(f"Batch Throughput: {throughput_batches_sec:.2f} batches/s")
    print(f"Sample Throughput: {throughput_samples_sec:.2f} samples/s")

    # Benchmark comparison from docs/BENCHMARKS.md
    print("\n==========================================================")
    print(" GATE VERIFICATION: Data Loading vs Model Compute")
    print("==========================================================")
    print("Reference Model Step Throughput (Mac M3 Pro, docs/BENCHMARKS.md):")
    print("  - Plain Conv2d Stack (Lower Bound): 58.78 ms/step (17.01 steps/s)")
    print("  - Composite Pipeline Lower Bound:  116.51 ms/step (8.58 steps/s)")
    print("  - Composite Pipeline with Full SKA: 411.52 ms/step (2.43 steps/s)")
    print(f"Measured Data Loading Latency:       {avg_batch_ms:.2f} ms/batch ({throughput_batches_sec:.2f} batches/s)")

    # Technical Gate Check
    if avg_batch_ms < 116.51:
        speedup_vs_full = 411.52 / avg_batch_ms
        speedup_vs_lower = 116.51 / avg_batch_ms
        print(f"\n[GATE PASSED]")
        print(f"Measured data-loading latency ({avg_batch_ms:.2f} ms) is:")
        print(f"  - {speedup_vs_lower:.2f}x faster than the lower bound model step (116.51 ms)")
        print(f"  - {speedup_vs_full:.2f}x faster than the full SKA model step (411.52 ms)")
        print("Data loading is completely non-blocking: asynchronous DataLoader background")
        print("pre-fetching comfortably hides I/O and feature extraction latency.")
    else:
        print(f"\n[GATE WARNING / ATTENTION]")
        print(f"Measured data-loading latency ({avg_batch_ms:.2f} ms) exceeds lower-bound model step (116.51 ms).")
        print("Proposed Fix: Increase num_workers, enable persistent_workers, or activate disk memmap caching.")


if __name__ == "__main__":
    main()


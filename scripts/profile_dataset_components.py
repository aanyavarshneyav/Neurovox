#!/usr/bin/env python3
"""Component-level profiling of dataset and dataloader throughput.

Measures:
1. Dataset __getitem__ components over 200 samples (single-process):
   - FLAC decode
   - Crop / Pad (180 frames = 29,040 samples)
   - Speed perturbation
   - Additive noise
   - Babble noise (including additional decodes)
   - Log-Mel filterbanks + CMN
   - SpecAugment
   - Total end-to-end __getitem__
2. DataLoader throughput (samples/sec) with num_workers in {2, 4, 8}.
3. DataLoader throughput with audio memmap cache enabled.
"""

from pathlib import Path
import sys
import time
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.cache import AudioMemmapWriter
from src.data.dataset import LibriSpeechDataset
from src.data.sampler import SpeakerBalancedBatchSampler
from src.data.transforms import (
    BabbleNoiseAugmentation,
    ColoredNoiseAugmentation,
    SpeedPerturbation,
    crop_or_pad_waveform,
)
from src.features.mfb import LogMelSpectrogram, SpecAugment, build_feature_extractor
from src.utils.paths import get_data_root, resolve_path


def profile_components(num_samples: int = 200) -> Dict[str, float]:
    """Measure single-process ms/sample for each pipeline component."""
    manifest_path = get_data_root() / "manifests" / "train-clean-100.csv"
    ds = LibriSpeechDataset(
        manifest_path=manifest_path,
        is_train=True,
        return_features=True,
        enable_speed_aug=True,
        enable_noise_aug=True,
        enable_babble_aug=True,
        enable_specaug=True,
        base_seed=42,
    )

    entries = ds.entries[:num_samples]
    target_samples = 29040  # 180 frames @ 16 kHz

    speed_aug = ds.speed_aug
    noise_aug = ds.noise_aug
    babble_aug = ds.babble_aug
    mel_cmn = build_feature_extractor(sample_rate=16000, n_mels=80, apply_cmn=True, enable_specaug=False)
    specaug = SpecAugment(freq_mask_param=10, time_mask_param=15, p=1.0)

    # Component timings
    times = {
        "flac_decode": 0.0,
        "speed_perturbation": 0.0,
        "crop_pad": 0.0,
        "noise_aug": 0.0,
        "babble_aug": 0.0,
        "log_mel_cmn": 0.0,
        "specaugment": 0.0,
        "total_getitem": 0.0,
    }

    print(f"Profiling dataset components over {num_samples} samples (single process)...")

    # 1. Individual component measurements
    for idx, entry in enumerate(entries):
        rng = ds._get_worker_rng(idx, epoch=1, draw_id=idx)

        # FLAC decode
        t0 = time.perf_counter()
        audio_path = resolve_path(entry["path"])
        wav_np, _ = sf.read(str(audio_path), dtype="float32")
        wav = torch.from_numpy(wav_np)
        times["flac_decode"] += time.perf_counter() - t0

        # Speed perturbation
        t0 = time.perf_counter()
        w_speed = speed_aug(wav.clone(), rng=rng)
        times["speed_perturbation"] += time.perf_counter() - t0

        # Crop / Pad
        t0 = time.perf_counter()
        w_crop = crop_or_pad_waveform(w_speed, target_samples=target_samples, rng=rng, is_train=True)
        times["crop_pad"] += time.perf_counter() - t0

        # Noise
        t0 = time.perf_counter()
        w_noise = noise_aug(w_crop.clone(), rng=rng)
        times["noise_aug"] += time.perf_counter() - t0

        # Babble
        t0 = time.perf_counter()
        w_babble = babble_aug(w_noise.clone(), rng=rng)
        times["babble_aug"] += time.perf_counter() - t0

        # Log-Mel + CMN
        t0 = time.perf_counter()
        mel = mel_cmn(w_babble.clone(), rng=rng)
        times["log_mel_cmn"] += time.perf_counter() - t0

        # SpecAugment
        t0 = time.perf_counter()
        _ = specaug(mel.clone(), rng=rng)
        times["specaugment"] += time.perf_counter() - t0

    # 2. End-to-end __getitem__
    t0_e2e = time.perf_counter()
    for idx in range(num_samples):
        _ = ds[(idx, 1, idx)]
    times["total_getitem"] = time.perf_counter() - t0_e2e

    results_ms = {k: (v / num_samples) * 1000.0 for k, v in times.items()}
    return results_ms


def profile_dataloader_throughput(
    worker_counts: List[int] = [2, 4, 8],
    num_batches: int = 50,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Measure DataLoader throughput (samples/sec) across different num_workers."""
    manifest_path = get_data_root() / "manifests" / "train-clean-100.csv"
    ds = LibriSpeechDataset(
        manifest_path=manifest_path,
        is_train=True,
        return_features=True,
        enable_speed_aug=True,
        enable_noise_aug=True,
        enable_babble_aug=True,
        enable_specaug=True,
        base_seed=42,
    )
    sampler = SpeakerBalancedBatchSampler(
        speaker_labels=ds.labels,
        batch_size=batch_size,
        batches_per_epoch=num_batches,
        seed=42,
    )

    throughput_results = {}

    for nw in worker_counts:
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=nw,
            persistent_workers=(nw > 0),
        )
        sampler.set_epoch(1)

        # Warmup 2 batches
        it = iter(loader)
        _ = next(it)
        _ = next(it)

        t0 = time.perf_counter()
        count = 0
        for _ in range(num_batches - 2):
            b = next(it)
            count += b[0].size(0)
        elapsed = time.perf_counter() - t0
        rate = count / max(elapsed, 1e-6)
        throughput_results[f"num_workers_{nw}"] = round(rate, 1)
        print(f"DataLoader num_workers={nw}: {count} samples in {elapsed:.2f}s ({rate:.1f} samples/s)")

    # Memmap cache test
    cache_dir = get_data_root() / "cache" / "train-50spk"
    manifest_50 = get_data_root() / "manifests" / "train-50spk.csv"
    if manifest_50.exists():
        if not (cache_dir / "audio_index.json").exists():
            print("Building audio memmap cache for 50-speaker subset...")
            writer = AudioMemmapWriter(cache_dir)
            writer.write_from_manifest(manifest_50, min_free_disk_gb=1.0)

        ds_cache = LibriSpeechDataset(
            manifest_path=manifest_50,
            cache_index_path=cache_dir / "audio_index.json",
            is_train=True,
            return_features=True,
            enable_speed_aug=True,
            enable_noise_aug=True,
            enable_babble_aug=True,
            enable_specaug=True,
            base_seed=42,
        )
        sampler_cache = SpeakerBalancedBatchSampler(
            speaker_labels=ds_cache.labels,
            batch_size=32,
            batches_per_epoch=num_batches,
            seed=42,
        )
        loader_cache = DataLoader(
            ds_cache,
            batch_sampler=sampler_cache,
            num_workers=2,
            persistent_workers=True,
        )
        sampler_cache.set_epoch(1)
        it = iter(loader_cache)
        _ = next(it)
        _ = next(it)
        t0 = time.perf_counter()
        count = 0
        for _ in range(num_batches - 2):
            b = next(it)
            count += b[0].size(0)
        elapsed = time.perf_counter() - t0
        rate = count / max(elapsed, 1e-6)
        throughput_results["memmap_cache_nw2"] = round(rate, 1)
        print(f"DataLoader with memmap cache (num_workers=2): {count} samples in {elapsed:.2f}s ({rate:.1f} samples/s)")

    return throughput_results


def main():
    print("=" * 70)
    print("B11 PROFILING: DATASET COMPONENTS & DATALOADER THROUGHPUT")
    print("=" * 70)

    comp_ms = profile_components(num_samples=200)
    print("\nComponent Timing (ms/sample over 200 samples):")
    for k, v in comp_ms.items():
        print(f"  {k:22s}: {v:6.2f} ms/sample")

    print("\nMeasuring DataLoader Throughput...")
    tp = profile_dataloader_throughput(worker_counts=[2, 4, 8], num_batches=30, batch_size=64)
    print("\nDataLoader Throughput Summary:")
    for k, v in tp.items():
        print(f"  {k:22s}: {v:6.1f} samples/s")

    return 0


if __name__ == "__main__":
    sys.exit(main())


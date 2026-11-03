"""PyTorch Dataset for LibriSpeech speaker verification with on-the-fly augmentations.

Implements the data loading, slicing, and augmentation pipeline specified in:
    J. Alam, A. Fathan, M. S. Alam, "A Hybrid Neural Approach to Speaker Verification
    with an Improved Additive Angular Margin Loss", IJCNN 2025, Section IV-A
    and docs/PROJECT_SPEC.md Table 1.1.

Pipeline features:
- Reads audio from disk or disk memmap cache with caching fallback.
- Random temporal slicing to exactly target_samples (29,040 samples = 180 frames @ 16 kHz)
  using wrap padding for short clips.
- Individual toggleable augmentations: speed perturbation, additive white/pink noise,
  in-batch multi-speaker babble noise, RIR reverberation, and SpecAugment.
- Deterministic seeding derived per-sample from (base_seed, epoch, idx, draw_id) via SeedSequence.
- Requires explicit (sample_idx, epoch, draw_id) in training mode (no silent fallback).
"""

import csv
import io
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.cache import AudioMemmapReader
from src.data.sampler import BatchItem
from src.data.transforms import (
    BabbleNoiseAugmentation,
    ColoredNoiseAugmentation,
    RIRReverberation,
    SpeedPerturbation,
    crop_or_pad_waveform,
)
from src.features.mfb import LogMelSpectrogram, build_feature_extractor
from src.utils.paths import resolve_path


class LibriSpeechDataset(Dataset):
    """PyTorch Dataset for LibriSpeech speaker verification."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        target_samples: int = 29040,
        is_train: bool = True,
        return_features: bool = True,
        return_dict: bool = False,
        enable_speed_aug: bool = False,
        enable_noise_aug: bool = False,
        enable_babble_aug: bool = False,
        enable_rir_aug: bool = False,
        enable_specaug: bool = True,
        speed_prob: float = 0.5,
        noise_prob: float = 0.3,
        babble_prob: float = 0.4,
        rir_prob: float = 0.5,
        specaug_p: float = 0.5,
        rir_dir: Optional[Union[str, Path]] = None,
        cache_index_path: Optional[Union[str, Path]] = None,
        base_seed: int = 42,
    ) -> None:
        """Initialize LibriSpeechDataset.

        Args:
            manifest_path: Path to manifest CSV file (e.g. data/manifests/train-clean-100.csv).
            target_samples: Target sequence length in samples (29,040 samples = 180 frames @ 16 kHz).
            is_train: True for training mode (random cropping & SpecAugment), False for evaluation.
            return_features: If True, extracts 80-dim log-Mel spectrogram; if False, returns waveform.
            return_dict: If True, returns dict with keys ('features'/'waveform', 'label', 'spk_id', 'utt_id');
                if False, returns standard tuple (data_tensor, label).
            enable_speed_aug: Whether to enable speed perturbation in training.
            enable_noise_aug: Whether to enable additive white/pink noise in training.
            enable_babble_aug: Whether to enable multi-speaker babble noise in training.
            enable_rir_aug: Whether to enable RIR reverberation (requires RIR files).
            enable_specaug: Whether to enable SpecAugment on log-Mel features in training.
            speed_prob: Probability of applying speed perturbation.
            noise_prob: Probability of applying additive colored noise.
            babble_prob: Probability of applying babble noise.
            rir_prob: Probability of applying RIR reverberation.
            specaug_p: Probability of applying SpecAugment.
            rir_dir: Optional path to directory containing RIR files.
            cache_index_path: Optional path to audio_index.json for disk memmap reading.
            base_seed: Base random seed for standalone execution.
        """
        self.manifest_path = resolve_path(manifest_path)
        self.target_samples = target_samples
        self.is_train = is_train
        self.return_features = return_features
        self.return_dict = return_dict
        self.base_seed = base_seed
        self.epoch = 0

        # Augmentation flags and parameters
        self.enable_speed_aug = enable_speed_aug
        self.enable_noise_aug = enable_noise_aug
        self.enable_babble_aug = enable_babble_aug
        self.enable_rir_aug = enable_rir_aug
        self.enable_specaug = enable_specaug

        self.speed_prob = speed_prob
        self.noise_prob = noise_prob
        self.babble_prob = babble_prob
        self.rir_prob = rir_prob
        self.specaug_p = specaug_p

        # Load manifest
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.entries: list[dict[str, Any]] = []
        with open(self.manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append(row)

        if len(self.entries) == 0:
            raise ValueError(f"Manifest {self.manifest_path} contains 0 utterances.")

        self.labels = [int(e["label"]) if e.get("label", "") != "" else -1 for e in self.entries]

        # Memmap reader setup
        self.memmap_reader = None
        if cache_index_path is not None:
            cpath = resolve_path(cache_index_path)
            if cpath.exists():
                self.memmap_reader = AudioMemmapReader(cpath)

        # Augmentation modules
        self.speed_aug = SpeedPerturbation(p=speed_prob) if enable_speed_aug and is_train else None
        self.noise_aug = ColoredNoiseAugmentation(p=noise_prob) if enable_noise_aug and is_train else None
        self.babble_aug = (
            BabbleNoiseAugmentation(
                p=babble_prob,
                audio_slice_loader=self._get_random_audio_slice,
            )
            if enable_babble_aug and is_train
            else None
        )
        self.rir_aug = RIRReverberation(rir_dir=rir_dir, p=rir_prob) if enable_rir_aug and is_train else None

        # Feature extractor: 80-dim log-Mel with CMN and SpecAugment via unified builder
        self.feature_extractor = build_feature_extractor(
            sample_rate=16000,
            n_mels=80,
            apply_cmn=True,
            enable_specaug=enable_specaug and is_train,
            specaug_p=specaug_p,
        )
        if not is_train:
            self.feature_extractor.eval()

    @property
    def augmentation_probabilities(self) -> Dict[str, float]:
        """Return the active augmentation probabilities."""
        return {
            "speed": self.speed_prob if self.speed_aug is not None else 0.0,
            "noise": self.noise_prob if self.noise_aug is not None else 0.0,
            "babble": self.babble_prob if self.babble_aug is not None else 0.0,
            "rir": self.rir_prob if self.rir_aug is not None else 0.0,
            "specaug": self.specaug_p if (self.enable_specaug and self.is_train) else 0.0,
        }

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic random seed progression across epochs."""
        self.epoch = epoch

    def _get_worker_rng(self, idx: int, epoch: int, draw_id: int) -> np.random.Generator:
        """Derive an independent NumPy RNG from (base_seed, epoch, sample_idx, draw_id)."""
        ss = np.random.SeedSequence([int(self.base_seed), int(epoch), int(idx), int(draw_id)])
        return np.random.default_rng(ss)

    def _read_raw_audio(self, entry: dict[str, Any]) -> np.ndarray:
        """Read audio either from memmap cache or on-the-fly from FLAC."""
        utt_id = entry["utt_id"]
        if self.memmap_reader is not None:
            return self.memmap_reader.read_audio(utt_id)

        audio_path = resolve_path(entry["path"])
        wav_data, _ = sf.read(str(audio_path), dtype="float32")
        return wav_data

    def _get_random_audio_slice(self, length: int, rng: np.random.Generator) -> torch.Tensor:
        """Fetch a random audio slice from a random utterance for babble noise mixing."""
        rand_idx = int(rng.integers(0, len(self.entries)))
        entry = self.entries[rand_idx]
        wav = torch.from_numpy(self._read_raw_audio(entry)).to(torch.float32)
        return crop_or_pad_waveform(wav, target_samples=length, rng=rng, is_train=True)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: Union[int, tuple, list, BatchItem]) -> Union[tuple[torch.Tensor, int], dict[str, Any]]:
        """Load, augment, crop, and optionally extract log-Mel features for sample idx.

        In training mode, idx MUST be a 3-element (sample_idx, epoch, draw_id) tuple/BatchItem.
        Plain integer indexing in train mode raises TypeError to prevent silent fallback.
        """
        if self.is_train:
            if not (isinstance(idx, (tuple, list, BatchItem)) and len(idx) == 3):
                raise TypeError(
                    f"In train mode, Dataset.__getitem__ requires a (sample_idx, epoch, draw_id) tuple/BatchItem "
                    f"from the batch sampler, got {type(idx).__name__}: {idx}"
                )
            sample_idx, epoch, draw_id = int(idx[0]), int(idx[1]), int(idx[2])
        else:
            if isinstance(idx, (tuple, list, BatchItem)):
                sample_idx = int(idx[0])
                epoch = int(idx[1]) if len(idx) > 1 else 0
                draw_id = int(idx[2]) if len(idx) > 2 else 0
            else:
                sample_idx = int(idx)
                epoch = 0
                draw_id = 0

        entry = self.entries[sample_idx]
        rng = self._get_worker_rng(sample_idx, epoch=epoch, draw_id=draw_id)

        # 1. Read raw audio float32
        audio_np = self._read_raw_audio(entry)
        waveform = torch.from_numpy(audio_np).to(torch.float32)

        # 2. Pre-crop to target_samples * 1.15 buffer if long, then apply speed perturbation
        if self.speed_aug is not None:
            buf_len = int(round(self.target_samples * 1.15))
            if waveform.size(-1) > buf_len:
                max_start = waveform.size(-1) - buf_len
                start = int(rng.integers(0, max_start + 1)) if self.is_train else (max_start // 2)
                waveform = waveform[..., start : start + buf_len]
            waveform = self.speed_aug(waveform, rng=rng)

        # 3. Slicing: random crop to target_samples (29,040 samples = 180 frames) with wrap padding
        waveform = crop_or_pad_waveform(
            waveform,
            target_samples=self.target_samples,
            rng=rng,
            is_train=self.is_train,
        )

        # 4. Waveform-level additive & reverberation augmentations
        if self.rir_aug is not None:
            waveform = self.rir_aug(waveform, rng=rng)
        if self.noise_aug is not None:
            waveform = self.noise_aug(waveform, rng=rng)
        if self.babble_aug is not None:
            waveform = self.babble_aug(waveform, rng=rng)

        # 5. Feature extraction: 80-dim log-Mel + CMN + SpecAugment
        if self.return_features:
            features = self.feature_extractor(waveform, rng=rng).squeeze(0)  # Shape: (80, 180)
            data_tensor = features
        else:
            data_tensor = waveform

        label = int(entry.get("label", -1)) if entry.get("label") != "" else -1

        if self.return_dict:
            return {
                "features" if self.return_features else "waveform": data_tensor,
                "label": label,
                "spk_id": entry.get("spk_id", ""),
                "utt_id": entry.get("utt_id", ""),
            }

        return data_tensor, label


__all__ = ["LibriSpeechDataset"]

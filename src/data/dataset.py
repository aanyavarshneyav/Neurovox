"""LibriSpeech PyTorch Dataset with on-the-fly acoustic feature extraction and augmentations.

Implements data loading, cropping, and waveform transformations per:
    J. Alam, A. Fathan, M. S. Alam, "A Hybrid Neural Approach to Speaker Verification
    with an Improved Additive Angular Margin Loss", IJCNN 2025, Section IV-A,
    docs/DATA_PROTOCOL.md Section 4 & 5, and docs/PROJECT_SPEC.md Table 1.1.

Key features:
1. Slicing: Random 180-frame crop (29,040 samples @ 16 kHz) with wrap padding for short clips.
2. Augmentation: Individually toggleable transforms (Speed, Colored Noise, Babble, RIR).
3. Seeding: All augmentations are strictly seeded using DataLoader worker seeds (worker_info.seed)
   combined with sample indices, guaranteeing zero hidden global state.
4. I/O: On-the-fly FLAC decoding via soundfile (default, optimal per BENCHMARKS.md) or optional
   int16 disk memmap reading via AudioMemmapReader when disk-cache execution is configured.
5. Features: Direct 80-dim log-Mel extraction with CMN and SpecAugment.
"""

import csv
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.data.cache import AudioMemmapReader
from src.data.transforms import (
    BabbleNoiseAugmentation,
    ColoredNoiseAugmentation,
    RIRReverberation,
    SpeedPerturbation,
    crop_or_pad_waveform,
)
from src.features.mfb import LogMelSpectrogram
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

        # Load manifest
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.entries: list[dict[str, Any]] = []
        with open(self.manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append(row)

        if not self.entries:
            raise ValueError(f"Manifest {self.manifest_path} contains 0 rows.")

        # Cache reader if specified
        self.memmap_reader: Optional[AudioMemmapReader] = None
        if cache_index_path is not None:
            resolved_cache_idx = resolve_path(cache_index_path)
            if resolved_cache_idx.exists():
                self.memmap_reader = AudioMemmapReader(resolved_cache_idx)

        # Build speaker metadata sequences
        self.speaker_ids: list[str] = [str(r["spk_id"]) for r in self.entries]
        self.labels: list[int] = [
            int(r["label"]) if "label" in r and r["label"] != "" else -1 for r in self.entries
        ]

        # Waveform augmentation modules
        self.speed_aug = SpeedPerturbation(p=0.5) if enable_speed_aug and is_train else None
        self.noise_aug = ColoredNoiseAugmentation(p=0.3) if enable_noise_aug and is_train else None
        self.babble_aug = (
            BabbleNoiseAugmentation(p=0.4, audio_slice_loader=self._load_random_speech_slice)
            if enable_babble_aug and is_train
            else None
        )
        self.rir_aug = RIRReverberation(rir_dir=rir_dir, p=0.5) if enable_rir_aug and is_train else None

        # Feature extractor: 80-dim log-Mel with CMN and SpecAugment
        self.feature_extractor = LogMelSpectrogram(
            sample_rate=16000,
            n_mels=80,
            apply_cmn=True,
            enable_specaug=enable_specaug and is_train,
            specaug_p=0.5,
        )
        if not is_train:
            self.feature_extractor.eval()

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic random seed progression across epochs."""
        self.epoch = epoch

    def _get_worker_rng(self, idx: int) -> np.random.Generator:
        """Derive an independent NumPy RNG from DataLoader worker seed and sample index."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            base = worker_info.seed
        else:
            base = self.base_seed
        # Hash combining base worker seed, epoch, and sample index
        seed = (base + self.epoch * 1_000_003 + idx * 7919) % (2**32)
        return np.random.default_rng(seed)

    def _read_raw_audio(self, entry: dict[str, Any]) -> np.ndarray:
        """Read audio either from memmap cache or on-the-fly from FLAC."""
        utt_id = entry["utt_id"]
        if self.memmap_reader is not None:
            return self.memmap_reader.read_audio(utt_id)

        # On-the-fly FLAC read via soundfile
        audio_path = resolve_path(entry["path"])
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio

    def _load_random_speech_slice(self, num_samples: int, rng: np.random.Generator) -> torch.Tensor:
        """Callback to load a random slice from another utterance for babble noise."""
        rand_idx = int(rng.integers(0, len(self.entries)))
        audio = self._read_raw_audio(self.entries[rand_idx])
        tensor = torch.from_numpy(audio)
        return crop_or_pad_waveform(tensor, target_samples=num_samples, rng=rng, is_train=True)

    def __len__(self) -> int:
        """Return total number of utterances in dataset."""
        return len(self.entries)

    def __getitem__(self, idx: int) -> Union[tuple[torch.Tensor, int], dict[str, Any]]:
        """Load, augment, crop, and optionally extract log-Mel features for sample idx.

        Args:
            idx: Dataset sample index.

        Returns:
            If return_dict is False:
                (features_or_waveform, label) tuple.
            If return_dict is True:
                Dict with 'features' or 'waveform', 'label', 'spk_id', 'utt_id'.
        """
        entry = self.entries[idx]
        rng = self._get_worker_rng(idx)

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
            features = self.feature_extractor(waveform).squeeze(0)  # Shape: (80, 180)
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


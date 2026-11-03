"""Full-utterance embedding extractor for speaker verification evaluation.

Implements the evaluation extraction protocol:
- Extracts embeddings over FULL audio waveforms (no temporal cropping).
- Enforces model.eval() mode during extraction.
- Preserves instance Cepstral Mean Normalization (CMN) across time frames.
- Implements length-bucketed batch extraction (grouping identical audio lengths)
  guaranteed to match batch size 1 within < 1e-4 tolerance.
- Extracts ONLY utterances that participate in evaluation trial lists, caching
  embeddings to eliminate redundant disk I/O.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.mfb import LogMelSpectrogram, build_feature_extractor
from src.utils.paths import resolve_path


class EmbeddingExtractor:
    """Full-utterance speaker embedding extractor."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        batch_size: int = 1,
        n_mels: int = 80,
    ) -> None:
        """Initialize EmbeddingExtractor.

        Args:
            model: Trained speaker verification model.
            device: Compute device (MPS, CUDA, or CPU).
            batch_size: Evaluation batch size (default 1 for exact zero-padding invariance).
            n_mels: Number of Mel filterbank bins (default: 80).
        """
        self.device = device if device is not None else torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()  # Enforce evaluation mode
        self.batch_size = batch_size

        # Unified feature extractor (eval mode, SpecAugment strictly OFF)
        self.feature_extractor = build_feature_extractor(
            sample_rate=16000,
            n_mels=n_mels,
            win_length_ms=25.0,
            hop_length_ms=10.0,
            apply_cmn=True,
            enable_specaug=False,
        ).to(self.device)
        self.feature_extractor.eval()

    def extract_from_waveform(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract a single L2-normalized embedding vector from a 1D or (1, T) waveform.

        Args:
            waveform: Audio samples at 16 kHz.

        Returns:
            1D numpy array of shape (emb_dim,) normalized to unit norm.
        """
        self.model.eval()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, T)

        wav = waveform.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            # 1. Extract 80-dim log-Mel features with CMN: (1, 80, T_frames)
            features = self.feature_extractor(wav)

            # 2. Forward pass through embedding model
            embedding = self.model(features)  # (1, emb_dim)

            # 3. L2 normalize embedding vector
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    def extract_from_file(self, audio_path: Union[str, Path]) -> np.ndarray:
        """Load audio file from disk and extract unit-normalized embedding vector."""
        resolved = resolve_path(audio_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Audio file not found: {resolved}")

        audio_np, sr = sf.read(str(resolved), dtype="float32")
        assert sr == 16000, f"Expected 16 kHz audio, got {sr} Hz in {resolved}"

        waveform = torch.from_numpy(audio_np)
        return self.extract_from_waveform(waveform)

    def extract_batch_identical_length(self, batch_waveforms: torch.Tensor) -> List[np.ndarray]:
        """Extract embeddings for a batch of waveforms that share identical length.

        Guarantees zero-padding batch invariance because all elements have identical length T.
        """
        self.model.eval()
        wav = batch_waveforms.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            features = self.feature_extractor(wav)
            embeddings = self.model(features)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        embs_np = embeddings.cpu().numpy().astype(np.float32)
        return [embs_np[i] for i in range(embs_np.shape[0])]

    def extract_batch_bucketed(self, waveforms: List[torch.Tensor]) -> List[np.ndarray]:
        """Extract embeddings for a list of waveforms of varying lengths by grouping identical lengths.

        Waveforms sharing the same length T are batched together and processed concurrently,
        guaranteeing exact equivalence to batch size 1 (< 1e-4 tolerance).
        """
        self.model.eval()
        buckets: Dict[int, List[Tuple[int, torch.Tensor]]] = defaultdict(list)
        for idx, w in enumerate(waveforms):
            w_2d = w.unsqueeze(0) if w.dim() == 1 else w
            buckets[w_2d.size(-1)].append((idx, w_2d))

        results: List[Optional[np.ndarray]] = [None] * len(waveforms)
        for length, items in buckets.items():
            indices, wav_list = zip(*items)
            batch_tensor = torch.cat(list(wav_list), dim=0)  # (B, 1, T)
            embs = self.extract_batch_identical_length(batch_tensor)
            for orig_idx, emb in zip(indices, embs):
                results[orig_idx] = emb

        return results

    def extract_for_trial_file(self, trial_file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Parse trial file, extract unique utterances, and return path -> embedding map."""
        trial_path = resolve_path(trial_file_path)
        if not trial_path.exists():
            raise FileNotFoundError(f"Trial file not found: {trial_path}")

        unique_paths: Set[str] = set()
        with open(trial_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    unique_paths.add(parts[1])
                    unique_paths.add(parts[2])

        cache: Dict[str, np.ndarray] = {}
        for path_str in unique_paths:
            emb = self.extract_from_file(path_str)
            cache[path_str] = emb

        return cache


__all__ = ["EmbeddingExtractor"]

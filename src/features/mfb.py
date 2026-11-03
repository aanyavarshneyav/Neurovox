"""Acoustic feature extraction: 80-dim log-Mel filterbanks with CMN and SpecAugment.

Implements the acoustic front-end specified in:
    J. Alam, A. Fathan, M. S. Alam, "A Hybrid Neural Approach to Speaker Verification
    with an Improved Additive Angular Margin Loss", IJCNN 2025, Section II, Section IV-A,
    and docs/PROJECT_SPEC.md Table 1.1.

Front-end specifications:
- 80-dimensional Mel filterbanks
- 25 ms window length (400 samples at 16 kHz), Hamming window
- 10 ms frame shift / hop length (160 samples at 16 kHz)
- Unpadded / uncentered STFT (center=False) yielding exactly 180 frames for 29,040 samples
- Instance mean normalization (Cepstral Mean Normalization, CMN) across time frames
- On-the-fly SpecAugment: frequency masking (F <= 10 bins) and time masking (T <= 15 frames),
  strictly applied during training only (bypassed in eval mode).
"""

from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T


class SpecAugment(nn.Module):
    """SpecAugment data augmentation applied to log-Mel spectrograms.

    Masks frequency channels (<= freq_mask_param bins) and time frames
    (<= time_mask_param frames) during training only, per Alam et al. (IJCNN 2025)
    and docs/DATA_PROTOCOL.md Section 5.4.

    When the module is in evaluation mode (self.training is False), this transform
    is an identity operation and no masking is applied.
    """

    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 15,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        p: float = 0.5,
    ) -> None:
        """Initialize SpecAugment module.

        Args:
            freq_mask_param: Maximum possible length of the frequency mask (F <= 10).
            time_mask_param: Maximum possible length of the time mask (T <= 15).
            num_freq_masks: Number of consecutive frequency masks to apply.
            num_time_masks: Number of consecutive time masks to apply.
            p: Probability of applying SpecAugment to an instance in training mode.
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p

        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param, iid_masks=True)

    def forward(
        self,
        spec: torch.Tensor,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        """Apply SpecAugment masking to spectrogram tensor.

        Args:
            spec: Spectrogram tensor of shape (B, F, T) or (F, T).
            rng: Optional NumPy random generator for deterministic worker-independent seeding.

        Returns:
            Masked spectrogram tensor with identical shape and dtype.
        """
        # Bypass during evaluation mode
        if not self.training:
            return spec

        if rng is not None:
            if self.p < 1.0 and float(rng.random()) >= self.p:
                return spec

            out = spec.clone()
            F_dim = out.size(-2)
            T_dim = out.size(-1)

            # Apply frequency masking
            for _ in range(self.num_freq_masks):
                f_len = int(rng.integers(0, self.freq_mask_param + 1))
                if f_len > 0 and F_dim > f_len:
                    f_start = int(rng.integers(0, F_dim - f_len + 1))
                    out[..., f_start : f_start + f_len, :] = 0.0

            # Apply time masking
            for _ in range(self.num_time_masks):
                t_len = int(rng.integers(0, self.time_mask_param + 1))
                if t_len > 0 and T_dim > t_len:
                    t_start = int(rng.integers(0, T_dim - t_len + 1))
                    out[..., :, t_start : t_start + t_len] = 0.0

            return out

        # Fallback to PyTorch transforms when rng is not provided
        if self.p < 1.0 and torch.rand(1, device=spec.device).item() >= self.p:
            return spec

        out = spec
        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            out = self.freq_mask(out)

        # Apply time masking
        for _ in range(self.num_time_masks):
            out = self.time_mask(out)

        return out


class LogMelSpectrogram(nn.Module):
    """80-dim log-Mel filterbank extractor with CMN and SpecAugment.

    Implements front-end extraction per Alam et al. (IJCNN 2025):
    - 16 kHz sampling rate
    - 25 ms window length (400 samples), Hamming window
    - 10 ms hop length (160 samples)
    - 80 Mel frequency bins
    - Instance mean normalization across the time dimension
    - SpecAugment (freq mask <= 10, time mask <= 15) in training only
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        win_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        n_fft: int = 400,
        apply_cmn: bool = True,
        enable_specaug: bool = True,
        freq_mask_param: int = 10,
        time_mask_param: int = 15,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        specaug_p: float = 0.5,
    ) -> None:
        """Initialize LogMelSpectrogram extractor.

        Args:
            sample_rate: Audio sampling rate in Hz (default 16000).
            n_mels: Number of Mel filterbank bins (default 80 per paper).
            win_length_ms: Analysis window length in milliseconds (default 25.0).
            hop_length_ms: Frame shift in milliseconds (default 10.0).
            n_fft: FFT size in samples (default 400).
            apply_cmn: Whether to apply instance mean normalization (default True).
            enable_specaug: Whether to include SpecAugment during training (default True).
            freq_mask_param: Maximum frequency mask width (default 10).
            time_mask_param: Maximum time mask width (default 15).
            num_freq_masks: Number of frequency masks (default 2).
            num_time_masks: Number of time masks (default 2).
            specaug_p: SpecAugment trigger probability in training mode (default 0.5).
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_length = int(round(sample_rate * (win_length_ms / 1000.0)))
        self.hop_length = int(round(sample_rate * (hop_length_ms / 1000.0)))
        self.n_fft = n_fft
        self.apply_cmn = apply_cmn
        self.enable_specaug = enable_specaug

        # torchaudio MelSpectrogram module
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            window_fn=torch.hamming_window,
            power=2.0,
            center=False,
        )

        # SpecAugment module
        self.specaug = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            num_freq_masks=num_freq_masks,
            num_time_masks=num_time_masks,
            p=specaug_p,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        """Compute 80-dim log-Mel filterbank spectrogram.

        Args:
            waveform: Raw audio tensor of shape (B, T_audio), (B, 1, T_audio),
                or 1D shape (T_audio,). Expected sampling rate is 16 kHz.
            rng: Optional NumPy random generator for deterministic SpecAugment.

        Returns:
            Normalized log-Mel filterbank tensor of shape (B, 80, T_frames),
            where T_frames = 180 for 29,040 samples.
        """
        # Ensure 2D tensor (B, T_audio)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        elif waveform.ndim != 2:
            raise ValueError(
                f"Expected waveform of shape (B, T), (B, 1, T), or (T,), got {waveform.shape}"
            )

        # Ensure float32 for stable numerics
        waveform = waveform.to(torch.float32)

        # Compute power Mel spectrogram: shape (B, n_mels, T_frames)
        mel = self.mel_transform(waveform)

        # Log compression with variance/numerical clamp
        log_mel = torch.log(torch.clamp(mel, min=1e-6))

        # Instance mean normalization (CMN) across the time dimension
        if self.apply_cmn:
            log_mel = log_mel - log_mel.mean(dim=-1, keepdim=True)

        # Apply SpecAugment in training mode if enabled
        if self.enable_specaug:
            log_mel = self.specaug(log_mel, rng=rng)

        return log_mel


def build_feature_extractor(
    sample_rate: int = 16000,
    n_mels: int = 80,
    win_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    n_fft: int = 400,
    apply_cmn: bool = True,
    enable_specaug: bool = False,
    freq_mask_param: int = 10,
    time_mask_param: int = 15,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
    specaug_p: float = 0.5,
) -> LogMelSpectrogram:
    """Unified factory function building LogMelSpectrogram front-end for dataset and extractor."""
    return LogMelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        win_length_ms=win_length_ms,
        hop_length_ms=hop_length_ms,
        n_fft=n_fft,
        apply_cmn=apply_cmn,
        enable_specaug=enable_specaug,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        num_freq_masks=num_freq_masks,
        num_time_masks=num_time_masks,
        specaug_p=specaug_p,
    )


def extract_log_mel(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    apply_cmn: bool = True,
) -> torch.Tensor:
    """Functional helper for deterministic inference log-Mel extraction.

    Args:
        waveform: Raw audio tensor of shape (B, T) or (T,).
        sample_rate: Audio sampling rate (default 16000).
        n_mels: Number of Mel bins (default 80).
        apply_cmn: Whether to apply instance mean normalization (default True).

    Returns:
        Log-Mel spectrogram tensor of shape (B, 80, T_frames).
    """
    extractor = LogMelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        apply_cmn=apply_cmn,
        enable_specaug=False,
    )
    extractor.eval()
    with torch.no_grad():
        return extractor(waveform)


__all__ = ["LogMelSpectrogram", "SpecAugment", "extract_log_mel", "build_feature_extractor"]


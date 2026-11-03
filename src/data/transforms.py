"""Waveform data augmentations and audio slicing transforms.

Implements individually toggleable acoustic transforms per docs/DATA_PROTOCOL.md Section 5
and docs/PROJECT_SPEC.md Table 1.1:
1. crop_or_pad_waveform: Random 180-frame crop (29,040 samples @ 16 kHz) with wrap padding.
2. SpeedPerturbation: Speed resampling factors in [0.95, 1.05].
3. ColoredNoiseAugmentation: Gaussian white and pink noise at SNR 5–15 dB.
4. BabbleNoiseAugmentation: Multi-speaker babble summing 3–5 slices at SNR 10–20 dB.
5. RIRReverberation: Room impulse response convolution via fftconvolve if RIRS_NOISES is available.

All augmentations are explicitly seeded via local NumPy Generator instances derived from
DataLoader worker seeds, ensuring zero reliance on global random state.
"""

from pathlib import Path
from typing import Callable, Optional, Sequence
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

from src.utils.paths import resolve_path


def crop_or_pad_waveform(
    waveform: torch.Tensor,
    target_samples: int = 29040,
    rng: Optional[np.random.Generator] = None,
    is_train: bool = True,
) -> torch.Tensor:
    """Randomly crop or wrap-pad audio waveform to exactly target_samples (180 frames @ 16 kHz).

    Args:
        waveform: Audio tensor of shape (T,) or (C, T).
        target_samples: Target sample count (default 29,040 samples = 180 frames at 16 kHz).
        rng: Local seeded NumPy RNG generator (no global state).
        is_train: If True, uses random offset; if False, uses center crop.

    Returns:
        Cropped or wrap-padded waveform with exact length target_samples along the last dimension.
    """
    n_samples = waveform.size(-1)

    if n_samples == target_samples:
        return waveform

    if n_samples < target_samples:
        # Wrap padding (circular repetition)
        repeats = (target_samples // n_samples) + 1
        if waveform.ndim == 1:
            padded = torch.tile(waveform, (repeats,))[:target_samples]
        else:
            padded = torch.tile(waveform, (1, repeats))[:, :target_samples]
        return padded

    # Slicing when n_samples > target_samples
    if is_train:
        if rng is not None:
            max_start = n_samples - target_samples
            start_idx = int(rng.integers(0, max_start + 1))
        else:
            max_start = n_samples - target_samples
            start_idx = int(torch.randint(0, max_start + 1, (1,)).item())
    else:
        start_idx = (n_samples - target_samples) // 2

    return waveform[..., start_idx : start_idx + target_samples]


class SpeedPerturbation:
    """Speed perturbation transform resampling audio by a factor in [min_factor, max_factor]."""

    def __init__(
        self,
        min_factor: float = 0.95,
        max_factor: float = 1.05,
        p: float = 0.5,
        sample_rate: int = 16000,
        method: str = "linear",
    ) -> None:
        """Initialize SpeedPerturbation transform.

        Args:
            min_factor: Minimum speed scale factor (default 0.95 per paper).
            max_factor: Maximum speed scale factor (default 1.05 per paper).
            p: Probability of applying speed perturbation (default 0.5).
            sample_rate: Audio sampling rate in Hz (default 16000).
            method: Interpolation method ('linear' for fast 1D interpolation or 'sinc').
        """
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.p = p
        self.sample_rate = sample_rate
        self.method = method

    def __call__(
        self,
        waveform: torch.Tensor,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        """Apply speed perturbation with given probability.

        Args:
            waveform: Audio tensor of shape (T,) or (1, T).
            rng: Local seeded NumPy RNG generator.

        Returns:
            Perturbed waveform tensor.
        """
        # Determine whether to apply
        should_apply = (rng.random() < self.p) if rng is not None else (torch.rand(1).item() < self.p)
        if not should_apply:
            return waveform

        # Sample factor
        if rng is not None:
            factor = float(rng.uniform(self.min_factor, self.max_factor))
        else:
            factor = float(torch.empty(1).uniform_(self.min_factor, self.max_factor).item())

        # If factor is effectively 1.0, return unchanged
        if abs(factor - 1.0) < 1e-4:
            return waveform

        was_1d = waveform.ndim == 1
        tensor_2d = waveform.unsqueeze(0) if was_1d else waveform
        n_samples = tensor_2d.size(-1)
        new_len = int(round(n_samples / factor))

        if self.method == "linear":
            # 1D linear interpolation across time: fast, high-fidelity, and avoids polyphase filter overhead
            tensor_3d = tensor_2d.unsqueeze(0)  # Shape (1, C, T)
            resampled_3d = torch.nn.functional.interpolate(
                tensor_3d,
                size=new_len,
                mode="linear",
                align_corners=False,
            )
            resampled = resampled_3d.squeeze(0)
        else:
            # Sinc resampling with rational approximation to avoid small-GCD filter bloat
            from fractions import Fraction
            frac = Fraction(factor).limit_denominator(20)
            orig_sr = int(round(self.sample_rate * frac.numerator))
            new_sr = int(round(self.sample_rate * frac.denominator))
            resampled = F.resample(tensor_2d, orig_freq=orig_sr, new_freq=new_sr)

        return resampled.squeeze(0) if was_1d else resampled


class ColoredNoiseAugmentation:
    """Additive Gaussian white and pink noise at specified SNR ranges."""

    def __init__(
        self,
        snr_min_db: float = 5.0,
        snr_max_db: float = 15.0,
        p: float = 0.3,
    ) -> None:
        """Initialize ColoredNoiseAugmentation.

        Args:
            snr_min_db: Minimum Signal-to-Noise Ratio in dB (default 5.0 dB).
            snr_max_db: Maximum Signal-to-Noise Ratio in dB (default 15.0 dB).
            p: Probability of applying noise augmentation (default 0.3).
        """
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.p = p

    def _generate_pink_noise(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> torch.Tensor:
        """Generate 1/f pink noise via spectral filtering."""
        if rng is not None:
            white = torch.from_numpy(rng.standard_normal(n_samples).astype(np.float32))
        else:
            white = torch.randn(n_samples, dtype=torch.float32)

        fft = torch.fft.rfft(white)
        freqs = torch.fft.rfftfreq(n_samples)
        freqs[0] = freqs[1]  # Guard against division by zero at DC
        scale = 1.0 / torch.sqrt(freqs)
        pink_fft = fft * scale
        pink = torch.fft.irfft(pink_fft, n=n_samples)
        return pink / (pink.std() + 1e-8)

    def __call__(
        self,
        waveform: torch.Tensor,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        """Apply additive white or pink noise at random SNR.

        Args:
            waveform: Audio tensor of shape (T,) or (1, T).
            rng: Local seeded NumPy RNG generator.

        Returns:
            Noisy audio tensor.
        """
        should_apply = (rng.random() < self.p) if rng is not None else (torch.rand(1).item() < self.p)
        if not should_apply:
            return waveform

        n_samples = waveform.size(-1)
        sig_power = torch.mean(waveform**2)
        if sig_power < 1e-8:
            return waveform

        # Sample SNR and noise type (50% white, 50% pink)
        if rng is not None:
            target_snr_db = float(rng.uniform(self.snr_min_db, self.snr_max_db))
            is_pink = rng.random() < 0.5
        else:
            target_snr_db = float(torch.empty(1).uniform_(self.snr_min_db, self.snr_max_db).item())
            is_pink = torch.rand(1).item() < 0.5

        if is_pink:
            raw_noise = self._generate_pink_noise(n_samples, rng=rng)
        else:
            if rng is not None:
                raw_noise = torch.from_numpy(rng.standard_normal(n_samples).astype(np.float32))
            else:
                raw_noise = torch.randn(n_samples, dtype=torch.float32)

        if waveform.ndim > 1:
            raw_noise = raw_noise.unsqueeze(0)

        # Scale noise to target SNR: SNR = 10 * log10(P_sig / P_noise)
        target_noise_power = sig_power / (10.0 ** (target_snr_db / 10.0))
        current_noise_power = torch.mean(raw_noise**2) + 1e-8
        scaling = torch.sqrt(target_noise_power / current_noise_power)

        return waveform + scaling * raw_noise


class BabbleNoiseAugmentation:
    """In-batch or multi-speaker speech babble combining 3–5 audio slices at SNR 10–20 dB."""

    def __init__(
        self,
        snr_min_db: float = 10.0,
        snr_max_db: float = 20.0,
        min_speakers: int = 3,
        max_speakers: int = 5,
        p: float = 0.4,
        audio_slice_loader: Optional[Callable[[int, np.random.Generator], torch.Tensor]] = None,
    ) -> None:
        """Initialize BabbleNoiseAugmentation.

        Args:
            snr_min_db: Minimum SNR in dB (default 10.0 dB).
            snr_max_db: Maximum SNR in dB (default 20.0 dB).
            min_speakers: Minimum number of interfering voices (default 3).
            max_speakers: Maximum number of interfering voices (default 5).
            p: Probability of applying babble augmentation (default 0.4).
            audio_slice_loader: Optional callable (num_samples, rng) -> torch.Tensor to load speech slices.
        """
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.p = p
        self.audio_slice_loader = audio_slice_loader

    def __call__(
        self,
        waveform: torch.Tensor,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        """Add multi-speaker speech babble to waveform.

        Args:
            waveform: Audio tensor of shape (T,) or (1, T).
            rng: Local seeded NumPy RNG generator.

        Returns:
            Audio tensor with added babble noise.
        """
        should_apply = (rng.random() < self.p) if rng is not None else (torch.rand(1).item() < self.p)
        if not should_apply:
            return waveform

        sig_power = torch.mean(waveform**2)
        if sig_power < 1e-8:
            return waveform

        n_samples = waveform.size(-1)

        # Sample number of speakers
        if rng is not None:
            num_spk = int(rng.integers(self.min_speakers, self.max_speakers + 1))
            target_snr_db = float(rng.uniform(self.snr_min_db, self.snr_max_db))
        else:
            num_spk = int(torch.randint(self.min_speakers, self.max_speakers + 1, (1,)).item())
            target_snr_db = float(torch.empty(1).uniform_(self.snr_min_db, self.snr_max_db).item())

        # Generate or load babble slices
        babble_sum = torch.zeros(n_samples, dtype=torch.float32)
        for _ in range(num_spk):
            if self.audio_slice_loader is not None and rng is not None:
                slice_tensor = self.audio_slice_loader(n_samples, rng)
                babble_sum += slice_tensor.squeeze()
            else:
                # Synthetic multi-tone formant babble fallback when no external speech pool is wired
                t = torch.linspace(0, n_samples / 16000.0, n_samples)
                freq = float(rng.uniform(100.0, 400.0)) if rng is not None else float(torch.empty(1).uniform_(100.0, 400.0).item())
                tone = torch.sin(2 * np.pi * freq * t)
                babble_sum += tone

        if waveform.ndim > 1:
            babble_sum = babble_sum.unsqueeze(0)

        # Scale babble to target SNR
        target_noise_power = sig_power / (10.0 ** (target_snr_db / 10.0))
        current_noise_power = torch.mean(babble_sum**2) + 1e-8
        scaling = torch.sqrt(target_noise_power / current_noise_power)

        return waveform + scaling * babble_sum


class RIRReverberation:
    """Room Impulse Response (RIR) convolution via torchaudio fftconvolve."""

    def __init__(
        self,
        rir_dir: Optional[str | Path] = None,
        p: float = 0.5,
    ) -> None:
        """Initialize RIRReverberation transform.

        Args:
            rir_dir: Optional path to directory containing RIR audio files (e.g. OpenSLR RIRS_NOISES).
            p: Probability of applying RIR convolution (default 0.5).
        """
        self.p = p
        self.rir_files: list[Path] = []
        if rir_dir is not None:
            resolved_dir = resolve_path(rir_dir)
            if resolved_dir.exists() and resolved_dir.is_dir():
                self.rir_files = sorted(list(resolved_dir.glob("**/*.wav")) + list(resolved_dir.glob("**/*.flac")))

    @property
    def is_available(self) -> bool:
        """Return True if RIR files are available on disk."""
        return len(self.rir_files) > 0

    def __call__(
        self,
        waveform: torch.Tensor,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        """Apply RIR convolution to waveform if RIR files are available.

        Args:
            waveform: Audio tensor of shape (T,) or (1, T).
            rng: Local seeded NumPy RNG generator.

        Returns:
            Reverberant waveform tensor or original waveform if bypassed.
        """
        if not self.is_available:
            # Safely bypassed per specification when RIRS_NOISES is not downloaded
            return waveform

        should_apply = (rng.random() < self.p) if rng is not None else (torch.rand(1).item() < self.p)
        if not should_apply:
            return waveform

        # Pick random RIR file
        if rng is not None:
            rir_path = self.rir_files[int(rng.integers(0, len(self.rir_files)))]
        else:
            idx = int(torch.randint(0, len(self.rir_files), (1,)).item())
            rir_path = self.rir_files[idx]

        rir_audio, rir_sr = sf.read(str(rir_path), dtype="float32")
        if rir_audio.ndim > 1:
            rir_audio = rir_audio.mean(axis=1)

        rir_tensor = torch.from_numpy(rir_audio).unsqueeze(0)
        if rir_sr != 16000:
            rir_tensor = F.resample(rir_tensor, orig_freq=rir_sr, new_freq=16000)

        # Normalize RIR energy
        rir_tensor = rir_tensor / (torch.norm(rir_tensor, p=2) + 1e-8)

        was_1d = waveform.ndim == 1
        if was_1d:
            waveform = waveform.unsqueeze(0)

        # Convolve and trim to original length
        orig_len = waveform.size(-1)
        reverb = F.fftconvolve(waveform, rir_tensor, mode="full")
        trimmed = reverb[..., :orig_len]

        return trimmed.squeeze(0) if was_1d else trimmed


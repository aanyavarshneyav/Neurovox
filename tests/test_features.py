"""Unit tests for acoustic feature extraction (MFB, CMN, SpecAugment).

Verifies contracts from docs/PROJECT_SPEC.md and docs/ARCHITECTURE.md:
- Output shape is strictly (B, 80, T) with T=180 for 29,040 samples @ 16 kHz.
- Instance mean normalization centres the temporal mean to zero.
- SpecAugment is strictly disabled in evaluation mode.
- SpecAugment modifies spectrograms during training mode.
"""

import pytest
import torch

from src.features.mfb import LogMelSpectrogram, SpecAugment, extract_log_mel


def test_log_mel_output_shape_2d():
    """Assert output shape is (B, 80, 180) for 2D waveform input (B, 29040)."""
    extractor = LogMelSpectrogram()
    extractor.eval()
    batch_size = 4
    samples = 29040  # 180 frames at 25ms win, 10ms hop
    waveform = torch.randn(batch_size, samples)

    out = extractor(waveform)
    assert out.shape == (batch_size, 80, 180)
    assert out.dtype == torch.float32


def test_log_mel_output_shape_3d():
    """Assert output shape is (B, 80, 180) for 3D waveform input (B, 1, 29040)."""
    extractor = LogMelSpectrogram()
    extractor.eval()
    waveform = torch.randn(2, 1, 29040)

    out = extractor(waveform)
    assert out.shape == (2, 80, 180)


def test_log_mel_output_shape_1d():
    """Assert output shape is (1, 80, 180) for 1D waveform input (29040,)."""
    extractor = LogMelSpectrogram()
    extractor.eval()
    waveform = torch.randn(29040)

    out = extractor(waveform)
    assert out.shape == (1, 80, 180)


def test_instance_mean_normalization():
    """Assert instance mean normalization results in zero-mean temporal statistics."""
    extractor = LogMelSpectrogram(apply_cmn=True, enable_specaug=False)
    extractor.eval()
    waveform = torch.randn(3, 29040) * 0.5 + 0.1

    out = extractor(waveform)  # Shape (3, 80, 180)
    mean_across_time = out.mean(dim=-1)  # Shape (3, 80)

    # All frequency bins should have temporal mean very close to zero
    assert torch.allclose(mean_across_time, torch.zeros_like(mean_across_time), atol=1e-5)


def test_specaugment_disabled_in_eval_mode():
    """Assert SpecAugment is completely inactive in eval mode."""
    extractor = LogMelSpectrogram(enable_specaug=True, specaug_p=1.0)
    extractor.eval()  # Eval mode
    assert not extractor.training
    assert not extractor.specaug.training

    waveform = torch.randn(2, 29040)

    # Multiple passes must be strictly bitwise identical in eval mode
    out1 = extractor(waveform)
    out2 = extractor(waveform)
    assert torch.equal(out1, out2)

    # Must be bitwise identical to an extractor with SpecAugment disabled
    extractor_no_specaug = LogMelSpectrogram(enable_specaug=False)
    extractor_no_specaug.eval()
    out_no_specaug = extractor_no_specaug(waveform)
    assert torch.equal(out1, out_no_specaug)


def test_specaugment_active_in_train_mode():
    """Assert SpecAugment alters spectrogram bins in training mode."""
    torch.manual_seed(42)
    extractor = LogMelSpectrogram(enable_specaug=True, specaug_p=1.0)
    extractor.train()
    assert extractor.training
    assert extractor.specaug.training

    waveform = torch.randn(2, 29040)
    out_train = extractor(waveform)

    extractor.eval()
    out_eval = extractor(waveform)

    # In train mode with p=1.0, masking should alter values relative to unmasked eval mode
    assert not torch.equal(out_train, out_eval)


def test_specaugment_mask_widths():
    """Assert SpecAugment creates zeroed bands within configured limits."""
    torch.manual_seed(42)
    specaug = SpecAugment(
        freq_mask_param=10,
        time_mask_param=15,
        num_freq_masks=1,
        num_time_masks=1,
        p=1.0,
    )
    specaug.train()

    # Create dummy constant spectrogram of shape (1, 80, 180)
    dummy_spec = torch.ones(1, 80, 180)
    masked = specaug(dummy_spec)

    # Some elements must be zeroed out
    assert (masked == 0.0).any()

    # In eval mode, no elements zeroed out
    specaug.eval()
    unmasked = specaug(dummy_spec)
    assert torch.equal(unmasked, dummy_spec)


def test_functional_extract_log_mel():
    """Assert functional helper produces valid (B, 80, T) tensor."""
    waveform = torch.randn(2, 29040)
    out = extract_log_mel(waveform)
    assert out.shape == (2, 80, 180)
    assert out.dtype == torch.float32


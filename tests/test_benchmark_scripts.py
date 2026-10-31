"""Unit tests for benchmark utility functions, layer shapes, and data transformations."""

import sys
from pathlib import Path
import pytest
import torch

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_env import get_hardware_info
from scripts.benchmark_hardware import (
    make_conv2d_stack,
    make_conv2d_stack_with_ska,
    make_conv1d_stack,
    LSTMWrapper,
    CompositeMockPipelineWithSKA,
)
from scripts.benchmark_dataloader import LibriSpeechFLACDataset


def test_hardware_info_keys():
    """Verify hardware info detection queries return valid system strings."""
    info = get_hardware_info()
    required_keys = [
        "os_version",
        "python_version",
        "torch_version",
        "torchaudio_version",
        "ram_gb",
        "free_disk_gb",
    ]
    for key in required_keys:
        assert key in info, f"Missing key {key} in get_hardware_info()"
        assert len(info[key]) > 0, f"Empty value for key {key}"


def test_conv2d_stack_shapes(dummy_tensor_shape):
    """Verify Conv2d frontend stack preserves temporal length and downsamples frequency."""
    B, _, F, T = dummy_tensor_shape
    C = 32
    stack = make_conv2d_stack(C)
    x = torch.randn(B, 1, F, T)
    out = stack(x)
    # Output frequency should be F // 4 = 20 bins, duration T = 180
    assert out.shape == (B, C, 20, T)


def test_conv2d_stack_with_ska_shapes(dummy_tensor_shape):
    """Verify Conv2d frontend stack with SKA ResBlocks preserves temporal length and downsamples frequency."""
    B, _, F, T = dummy_tensor_shape
    C = 32
    stack = make_conv2d_stack_with_ska(C)
    x = torch.randn(B, 1, F, T)
    out = stack(x)
    assert out.shape == (B, C, 20, T)


def test_composite_mock_pipeline_with_ska_shapes(dummy_tensor_shape):
    """Verify full composite pipeline mock with SKA branches and MLASP produces valid logits."""
    B, _, F, T = dummy_tensor_shape
    model = CompositeMockPipelineWithSKA(c=32, c1=256)
    x = torch.randn(B, 1, F, T)
    out = model(x)
    assert out.shape == (B, 251)


def test_conv1d_stack_shapes():
    """Verify TDNN Conv1d layer matches expected scaled tensor dimensions."""
    B, T = 2, 180
    in_channels = 1920  # 3 * d_c where d_c = 640
    out_channels = 256  # C_1
    layer = make_conv1d_stack(in_channels, out_channels)
    x = torch.randn(B, in_channels, T)
    out = layer(x)
    assert out.shape == (B, out_channels, T)


def test_lstm_wrapper_shapes():
    """Verify LSTM layer processes sequence over time and preserves T=180."""
    B, T, C1 = 2, 180, 256
    lstm = LSTMWrapper(C1)
    x = torch.randn(B, T, C1)
    out = lstm(x)
    assert out.shape == (B, T, C1)


def test_flac_dataset_mel_spectrogram_shape():
    """Verify Mel-spectrogram transformation produces exact (1, 80, 180) tensor from 29,040 samples."""
    dataset = LibriSpeechFLACDataset([])
    waveform = torch.randn(1, 29040)
    mel = dataset.mel_transform(waveform)
    log_mel = torch.log(torch.clamp(mel, min=1e-6))
    assert log_mel.shape == (1, 80, 180)


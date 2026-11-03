"""Unit tests for ECAPA-TDNN baseline architecture (Phase 2C).

Tests:
1. Input/output tensor contracts for (B, 80, T) and (B, 1, 80, T).
2. Forward and backward gradient propagation.
3. Parameter count audits via torchinfo (parameter-matched and standard width).
4. Embedding extractor integration and L2-normalization.
5. Device compatibility (CPU and MPS).
6. Evaluation mode batch size invariance.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F
from torchinfo import summary

from src.eval.extractor import EmbeddingExtractor
from src.models.ecapa_tdnn import (
    ECAPATDNN,
    ecapa_tdnn_param_matched,
    ecapa_tdnn_standard,
)


def test_ecapa_tdnn_output_shapes_and_variable_t():
    """Verify ECAPA-TDNN produces (B, 192) embeddings across variable sequence lengths."""
    model = ecapa_tdnn_param_matched(in_features=80, emb_dim=192)
    model.eval()

    for t_len in [100, 180, 250]:
        # Test 3D (B, 80, T)
        x3d = torch.randn(2, 80, t_len)
        with torch.no_grad():
            out3d = model(x3d)
        assert out3d.shape == (2, 192), f"Expected (2, 192) for T={t_len}, got {out3d.shape}"

        # Test 4D (B, 1, 80, T)
        x4d = torch.randn(2, 1, 80, t_len)
        with torch.no_grad():
            out4d = model(x4d)
        assert out4d.shape == (2, 192), f"Expected (2, 192) for T={t_len}, got {out4d.shape}"


def test_ecapa_tdnn_forward_backward():
    """Verify gradients propagate to all trainable parameters without NaNs or Infs."""
    model = ecapa_tdnn_param_matched(in_features=80, emb_dim=192)
    model.train()

    x = torch.randn(2, 80, 100, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients"
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradients"


def test_ecapa_tdnn_parameter_audit_torchinfo():
    """Verify exact parameter counts with torchinfo.summary().

    Target SKA-HNN parameter count: 4,167,700.
    Parameter-matched ECAPA-TDNN (C=400, MFA=1200):
        Measured total: 4,123,210 params (diff: -44,490, -1.07%).
    Standard ECAPA-TDNN (C=1024, MFA=3072):
        Measured total: 20,767,552 params.
    """
    # 1. Parameter-matched
    model_matched = ecapa_tdnn_param_matched(in_features=80, emb_dim=192)
    s_matched = summary(model_matched, input_size=(1, 80, 180), verbose=0)

    assert s_matched.total_params == 4_123_210, (
        f"Parameter-matched total params mismatch: got {s_matched.total_params:,}, expected 4,123,210"
    )
    assert s_matched.trainable_params == 4_123_210
    assert s_matched.total_params - s_matched.trainable_params == 0

    ska_hnn_params = 4_167_700
    param_delta = abs(s_matched.total_params - ska_hnn_params)
    relative_delta = param_delta / ska_hnn_params
    assert relative_delta < 0.02, (
        f"Parameter match deviation {relative_delta*100:.2f}% exceeds 2% margin"
    )

    # 2. Standard ECAPA (C=1024)
    model_std = ecapa_tdnn_standard(in_features=80, emb_dim=192)
    s_std = summary(model_std, input_size=(1, 80, 180), verbose=0)
    assert s_std.total_params == 20_767_552, (
        f"Standard ECAPA total params mismatch: got {s_std.total_params:,}, expected 20,767,552"
    )


def test_ecapa_tdnn_extractor_integration():
    """Verify EmbeddingExtractor processes audio through ECAPATDNN and produces unit-norm embeddings."""
    model = ecapa_tdnn_param_matched(in_features=80, emb_dim=192)
    extractor = EmbeddingExtractor(model=model, device=torch.device("cpu"), batch_size=1)

    # Synthetic 2-second audio waveform at 16 kHz
    wav = torch.randn(32000)
    emb = extractor.extract_from_waveform(wav)

    assert emb.shape == (192,)
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


def test_ecapa_tdnn_device_compatibility():
    """Verify forward and backward execution on CPU and MPS."""
    devices = [torch.device("cpu")]
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    for dev in devices:
        model = ecapa_tdnn_param_matched(in_features=80, emb_dim=192).to(dev)
        model.train()

        x = torch.randn(2, 80, 100, device=dev, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 192)
        assert out.device.type == dev.type

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


def test_ecapa_tdnn_eval_batch_invariance():
    """Verify evaluation embeddings are identical regardless of batch size."""
    model = ecapa_tdnn_param_matched(in_features=80, emb_dim=192)
    model.eval()

    torch.manual_seed(42)
    x0 = torch.randn(1, 80, 150)
    x1 = torch.randn(1, 80, 150)
    x_batch = torch.cat([x0, x1], dim=0)

    with torch.no_grad():
        emb_single = model(x0)
        emb_batched = model(x_batch)[:1]

    diff = (emb_single - emb_batched).abs().max().item()
    assert diff < 1e-5, f"Batch invariance violated: max difference = {diff:.2e}"

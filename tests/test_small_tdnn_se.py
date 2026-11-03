"""Unit tests for small_tdnn_se model architecture."""

import pytest
import torch
from torchinfo import summary

from src.models.small_tdnn_se import SmallTDNNSE, small_tdnn_se


def test_small_tdnn_se_shapes():
    """Verify small_tdnn_se outputs strictly (B, 192) for arbitrary batch size and duration T."""
    model = small_tdnn_se(in_features=80, channels=128, emb_dim=192)
    model.eval()

    for B in [1, 4]:
        for T in [100, 180, 350]:
            x = torch.randn(B, 80, T)
            out = model(x)
            assert out.shape == (B, 192), f"Expected shape ({B}, 192), got {out.shape}"
            assert not torch.isnan(out).any(), "Output contains NaN"

    # Also test with 4D input (B, 1, 80, T)
    x4d = torch.randn(2, 1, 80, 180)
    out4d = model(x4d)
    assert out4d.shape == (2, 192)


def test_small_tdnn_se_forward_backward():
    """Verify small_tdnn_se produces finite, non-zero gradients."""
    model = small_tdnn_se(in_features=80, channels=128, emb_dim=192)
    model.train()

    x = torch.randn(4, 80, 180, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input gradient is None"
    assert not torch.isnan(x.grad).any(), "Input gradient has NaN"
    assert x.grad.norm().item() > 0, "Input gradient is zero"

    # Verify model parameters received gradient
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None on {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient has NaN on {name}"


def test_small_tdnn_se_parameter_summary():
    """Audit parameter count of small_tdnn_se via torchinfo."""
    model = small_tdnn_se(in_features=80, channels=128, emb_dim=192)
    stats = summary(model, input_size=(1, 80, 180), verbose=0)
    total_params = stats.total_params
    trainable_params = stats.trainable_params

    assert total_params > 0, "Total params must be > 0"
    assert trainable_params == total_params, "All params should be trainable"
    print(f"\nMeasured small_tdnn_se total parameters: {total_params:,} ({total_params / 1e6:.3f}M)")


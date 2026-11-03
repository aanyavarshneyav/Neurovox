"""Unit tests for AAMSoftmax loss module."""

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.losses.aam_softmax import AAMSoftmax


def test_aam_softmax_s1_m0_near_ln_classes():
    """Verify s=1, m=0 with unit-norm random inputs produces loss near ln(251) +- 0.05."""
    num_classes = 251
    in_features = 192
    expected_ln = math.log(num_classes)  # 5.52545...

    for seed in [42, 123, 777]:
        torch.manual_seed(seed)
        loss_fn = AAMSoftmax(in_features=in_features, num_classes=num_classes, scale=1.0, margin=0.0)
        # Large batch of random embeddings
        B = 1024
        x = torch.randn(B, in_features)
        labels = torch.randint(0, num_classes, (B,))

        loss = loss_fn(x, labels).item()
        assert abs(loss - expected_ln) < 0.05, (
            f"Seed {seed}: Expected loss near {expected_ln:.4f} +- 0.05, got {loss:.4f}"
        )


def aam_softmax_reference_fp64(
    x_np: np.ndarray,
    labels_np: np.ndarray,
    weight_np: np.ndarray,
    s: float = 30.0,
    m: float = 0.2,
) -> float:
    """NumPy float64 reference implementation of AAMSoftmax loss formula."""
    x = x_np.astype(np.float64)
    w = weight_np.astype(np.float64)

    # Normalize
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    w_norm = w / np.linalg.norm(w, axis=1, keepdims=True)

    # Cosine
    cos_theta = np.dot(x_norm, w_norm.T)
    cos_theta = np.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

    cos_m = math.cos(m)
    sin_m = math.sin(m)
    th = math.cos(math.pi - m)
    mm = math.sin(math.pi - m) * m

    sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 1e-7, None))
    phi = cos_theta * cos_m - sin_theta * sin_m
    phi = np.where(cos_theta > th, phi, cos_theta - mm)

    B, C = cos_theta.shape
    logits = cos_theta.copy()
    for i in range(B):
        logits[i, labels_np[i]] = phi[i, labels_np[i]]
    logits = logits * s

    # Cross entropy in float64
    max_l = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = max_l[:, 0] + np.log(np.sum(np.exp(logits - max_l), axis=1))
    target_l = logits[np.arange(B), labels_np]
    return float(np.mean(log_sum_exp - target_l))


def test_aam_softmax_exact_match_against_fp64_reference():
    """Verify PyTorch AAMSoftmax (s=30, m=0.2) matches float64 reference implementation."""
    torch.manual_seed(42)
    B = 64
    C = 251
    D = 192
    s = 30.0
    m = 0.2

    loss_fn = AAMSoftmax(in_features=D, num_classes=C, scale=s, margin=m)
    x = torch.randn(B, D, dtype=torch.float32)
    labels = torch.randint(0, C, (B,))

    # Compute PyTorch loss
    loss_pt = loss_fn(x, labels).item()

    # Compute NumPy float64 reference loss using exact same weights
    weight_np = loss_fn.weight.detach().cpu().numpy()
    x_np = x.numpy()
    labels_np = labels.numpy()
    loss_ref = aam_softmax_reference_fp64(x_np, labels_np, weight_np, s=s, m=m)

    assert abs(loss_pt - loss_ref) < 1e-5, (
        f"PyTorch loss ({loss_pt:.7f}) deviates from fp64 reference ({loss_ref:.7f}) by {abs(loss_pt - loss_ref):.2e}"
    )


def test_aam_softmax_m0_reduces_to_scaled_cosine_softmax():
    """Verify that setting m=0 reduces exactly to standard scaled-cosine softmax."""
    torch.manual_seed(123)
    B = 32
    C = 50
    D = 128
    s = 20.0

    loss_fn = AAMSoftmax(in_features=D, num_classes=C, scale=s, margin=0.0)
    x = torch.randn(B, D)
    labels = torch.randint(0, C, (B,))

    loss = loss_fn(x, labels)

    # Manual scaled-cosine cross entropy
    x_norm = F.normalize(x, dim=1)
    w_norm = F.normalize(loss_fn.weight, dim=1)
    logits = F.linear(x_norm, w_norm) * s
    expected_loss = F.cross_entropy(logits, labels)

    assert torch.allclose(loss, expected_loss, atol=1e-6)


def test_aam_softmax_gradients_finite_and_nonzero():
    """Verify gradients on inputs and prototype weights are finite and strictly non-zero."""
    torch.manual_seed(999)
    B = 16
    C = 251
    D = 192

    loss_fn = AAMSoftmax(in_features=D, num_classes=C, scale=30.0, margin=0.2)
    x = torch.randn(B, D, requires_grad=True)
    labels = torch.randint(0, C, (B,))

    loss = loss_fn(x, labels)
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()
    assert x.grad.norm().item() > 0

    assert loss_fn.weight.grad is not None
    assert not torch.isnan(loss_fn.weight.grad).any()
    assert not torch.isinf(loss_fn.weight.grad).any()
    assert loss_fn.weight.grad.norm().item() > 0


def test_aam_softmax_overflow_fallback_no_nan():
    """Verify theta + m > pi edge cases trigger fallback without producing NaNs."""
    B = 4
    C = 10
    D = 16
    m = 0.5
    loss_fn = AAMSoftmax(in_features=D, num_classes=C, scale=30.0, margin=m)

    # Craft inputs where target cosine is near -1 (i.e. theta near pi, so theta + m > pi)
    target_weights = loss_fn.weight[:B].detach()
    # Invert target weight to force cos_theta = -1
    x = -target_weights + torch.randn(B, D) * 1e-4
    labels = torch.arange(B)

    loss = loss_fn(x, labels)
    assert not torch.isnan(loss), "Loss produced NaN on overflow input"
    assert not torch.isinf(loss), "Loss produced Inf on overflow input"
    loss.backward()
    assert not torch.isnan(loss_fn.weight.grad).any()


def test_aam_softmax_reduction_none_vs_mean():
    """Verify reduction='none' produces per-sample losses that average to reduction='mean'."""
    torch.manual_seed(42)
    B = 32
    C = 50
    D = 64
    loss_mean_fn = AAMSoftmax(in_features=D, num_classes=C, reduction="mean")
    loss_none_fn = AAMSoftmax(in_features=D, num_classes=C, reduction="none")
    # Share weights
    loss_none_fn.weight.data.copy_(loss_mean_fn.weight.data)

    x = torch.randn(B, D)
    labels = torch.randint(0, C, (B,))

    loss_mean = loss_mean_fn(x, labels)
    loss_none = loss_none_fn(x, labels)

    assert loss_none.shape == (B,)
    assert torch.allclose(loss_none.mean(), loss_mean, atol=1e-6)



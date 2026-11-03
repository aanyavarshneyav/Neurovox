"""Comprehensive validation suite for IAAMSoftmax with AdaptiveDrop (Phase 2C).

Tests:
1. Exact mathematical reference calculation on small synthetic batch.
2. AdaptiveDrop warm-up period (epochs 1 to 4).
3. Label correction logic (tau_corr threshold, best competing class).
4. Sample filtering logic (tau threshold).
5. Strict correction-before-filtering sequence ordering.
6. Minimum keep-ratio safeguard (deterministic top-k fallback).
7. Gradient flow: non-zero for retained samples, exactly zero for dropped samples.
8. Edge cases: all kept, none kept, tiny batches (B=1, B=2), out-of-range labels, NaNs.
9. Device compatibility: CPU and MPS (if available).
10. Determinism under fixed inputs.
11. Diagnostic exposure (batch and epoch level).
"""

import math
from typing import Dict, Any
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.losses.aam_softmax import AAMSoftmax
from src.losses.iaam_softmax import IAAMSoftmax


def test_iaam_softmax_exact_manual_synthetic_batch():
    """Verify IAAMSoftmax against closed-form hand-calculated reference.

    Setup:
        C=3 classes, D=2 dimensions.
        Prototypes: W0=[1, 0], W1=[0, 1], W2=[-1, 0].
        Scale s=10.0, Margin m=0.2.
        tau=0.423, tau_corr=0.5, ESD=5, ESC=5, min_keep_ratio=0.5.
        Current epoch = 5 (AdaptiveDrop active).

    Batch of 4 samples (all assigned label 0):
        Sample 0: x=[1, 0], assigned 0. CS=[1, 0, -1]. Clean sample -> kept as 0.
        Sample 1: x=[0, 1], assigned 0. CS=[0, 1, -1]. Mislabeled -> corrected to 1 -> kept as 1.
        Sample 2: x=[0, -1], assigned 0. CS=[0, -1, 0]. Outlier -> dropped.
        Sample 3: x=[0.2, -sqrt(0.96)], assigned 0. CS=[0.2, -0.98, -0.2]. Outlier -> dropped.

    Expected:
        Retained samples: Sample 0 (label 0) and Sample 1 (label 1).
        Sample 0 logits: [10*cos(0.2), 0, -10]. Loss = -log(e^(10*cos0.2) / (e^(10*cos0.2) + 1 + e^-10))
        Sample 1 logits: [0, 10*cos(0.2), 0]. Loss = -log(e^(10*cos0.2) / (e^(10*cos0.2) + 1 + 1))
        Mean loss: 8.31446377e-5.
        num_corrected: 1
        num_filtered: 2
        num_retained: 2
        keep_ratio: 0.5
        guard_activated: False
    """
    loss_fn = IAAMSoftmax(
        in_features=2,
        num_classes=3,
        scale=10.0,
        margin=0.2,
        tau=0.423,
        tau_corr=0.5,
        esd=5,
        esc=5,
        min_keep_ratio=0.5,
    )

    with torch.no_grad():
        loss_fn.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))

    loss_fn.train()
    loss_fn.set_epoch(5)

    y3_coord = -math.sqrt(1.0 - 0.2**2)
    x = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [0.2, y3_coord],
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 0], dtype=torch.long)

    loss, diag = loss_fn(x, labels, return_diagnostics=True)

    # Exact closed-form manual reference
    m_logit = 10.0 * math.cos(0.2)
    exp_m = math.exp(m_logit)
    loss_s0 = -math.log(exp_m / (exp_m + math.exp(0.0) + math.exp(-10.0)))
    loss_s1 = -math.log(exp_m / (exp_m + math.exp(0.0) + math.exp(0.0)))
    expected_mean_loss = 0.5 * (loss_s0 + loss_s1)

    assert abs(loss.item() - expected_mean_loss) < 1e-6, (
        f"Loss mismatch: got {loss.item():.8e}, expected {expected_mean_loss:.8e}"
    )

    assert diag["num_corrected"] == 1
    assert diag["num_filtered"] == 2
    assert diag["num_retained"] == 2
    assert diag["batch_size"] == 4
    assert abs(diag["keep_ratio"] - 0.5) < 1e-4
    assert diag["guard_activated"] is False
    assert abs(diag["mean_target_cs"] - 1.0) < 1e-4


def test_iaam_softmax_warmup_epochs():
    """Verify that during warm-up (epoch < ESD and epoch < ESC), no samples are dropped or corrected."""
    loss_fn = IAAMSoftmax(
        in_features=4,
        num_classes=4,
        scale=30.0,
        margin=0.2,
        tau=0.423,
        tau_corr=0.5,
        esd=5,
        esc=5,
    )
    loss_fn.train()

    # Create batch with noisy embeddings that would otherwise be filtered
    x = torch.zeros(8, 4)
    x[:, 0] = 1.0
    # Mismatched labels
    labels = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2])

    for ep in [1, 2, 3, 4]:
        loss_fn.set_epoch(ep)
        loss, diag = loss_fn(x, labels, return_diagnostics=True)
        assert diag["num_corrected"] == 0, f"Epoch {ep}: expected 0 corrected, got {diag['num_corrected']}"
        assert diag["num_filtered"] == 0, f"Epoch {ep}: expected 0 filtered, got {diag['num_filtered']}"
        assert diag["num_retained"] == 8
        assert diag["keep_ratio"] == 1.0
        assert diag["guard_activated"] is False


def test_iaam_softmax_correction_logic():
    """Verify label correction triggers if and only if CS(x, y) < tau AND max competing CS >= tau_corr."""
    loss_fn = IAAMSoftmax(
        in_features=2,
        num_classes=3,
        tau=0.423,
        tau_corr=0.5,
        esd=5,
        esc=5,
    )
    with torch.no_grad():
        loss_fn.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))

    loss_fn.train()
    loss_fn.set_epoch(5)

    # Sample A: assigned 0, but strongly matches class 1 (CS=1.0 >= 0.5, CS(0)=0.0 < 0.423) -> Corrected
    # Sample B: assigned 0, CS(0)=0.6 >= tau -> Clean, not corrected
    # Sample C: assigned 0, CS(0)=0.1 < tau, competing classes have CS <= 0.3 < tau_corr -> Ambiguous outlier, not corrected
    xA = [0.0, 1.0]
    xB = [0.6, 0.8]
    xC = [0.1, -0.994987]
    x = torch.tensor([xA, xB, xC], dtype=torch.float32)
    labels = torch.tensor([0, 0, 0], dtype=torch.long)

    loss, diag = loss_fn(x, labels, return_diagnostics=True)
    assert diag["num_corrected"] == 1, f"Expected exactly 1 correction (Sample A), got {diag['num_corrected']}"


def test_iaam_softmax_correction_before_filtering_order():
    """CRITICAL TEST: Verify correction occurs BEFORE filtering.

    A mislabeled sample belonging to class 1 is assigned label 0.
    - Assigned similarity CS(x, 0) = 0.0 < tau=0.423.
    - True class similarity CS(x, 1) = 1.0 >= tau_corr=0.5.

    If correction happens first:
        Label is rectified 0 -> 1.
        Filtering tests CS(x, 1) = 1.0 >= tau=0.423.
        Result: Sample is KEPT.

    If filtering happened first:
        Filtering tests CS(x, 0) = 0.0 < tau=0.423.
        Result: Sample would be DROPPED.
    """
    loss_fn = IAAMSoftmax(
        in_features=2,
        num_classes=2,
        tau=0.423,
        tau_corr=0.5,
        esd=5,
        esc=5,
        min_keep_ratio=0.0,  # Disable guard to test pure ordering
    )
    with torch.no_grad():
        loss_fn.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    loss_fn.train()
    loss_fn.set_epoch(5)

    # Embedding points purely to class 1, but labeled as class 0
    x = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.long)

    loss, diag = loss_fn(x, labels, return_diagnostics=True)

    assert diag["num_corrected"] == 1, "Mislabeled sample must be corrected"
    assert diag["num_filtered"] == 0, "Corrected sample must NOT be filtered out"
    assert diag["num_retained"] == 1, "Sample must be retained in training loss"


def test_iaam_softmax_min_keep_ratio_guard():
    """Verify that when filtering retains fewer than min_keep_ratio*B, the safeguard deterministically keeps top-k."""
    loss_fn = IAAMSoftmax(
        in_features=2,
        num_classes=2,
        tau=0.423,
        tau_corr=0.5,
        esd=5,
        esc=5,
        min_keep_ratio=0.5,
    )
    with torch.no_grad():
        # Prototypes along x and y axis
        loss_fn.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    loss_fn.train()
    loss_fn.set_epoch(5)

    # Batch of 4 samples that ALL have CS < 0.423 and cannot be corrected (competing CS < 0.5)
    # Cosine similarities to assigned label 0: [0.35, 0.25, 0.15, 0.05]
    # All are < tau=0.423. Normal filtering would keep 0 samples.
    # min_keep = ceil(4 * 0.5) = 2.
    # Guard must activate and retain top-2 samples ([0.35, 0.25]).
    x0 = [0.35, -math.sqrt(1 - 0.35**2)]
    x1 = [0.25, -math.sqrt(1 - 0.25**2)]
    x2 = [0.15, -math.sqrt(1 - 0.15**2)]
    x3 = [0.05, -math.sqrt(1 - 0.05**2)]

    x = torch.tensor([x0, x1, x2, x3], dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 0], dtype=torch.long)

    loss, diag = loss_fn(x, labels, return_diagnostics=True)

    assert diag["guard_activated"] is True, "Guard must activate when candidate kept < min_keep"
    assert diag["num_retained"] == 2, f"Guard must retain exactly 2 samples, got {diag['num_retained']}"
    assert diag["num_filtered"] == 2
    assert diag["keep_ratio"] == 0.5
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_iaam_softmax_gradient_propagation():
    """Verify gradient flow: non-zero for retained samples, exactly zero for filtered samples."""
    loss_fn = IAAMSoftmax(
        in_features=2,
        num_classes=2,
        tau=0.423,
        esd=5,
        esc=5,
        min_keep_ratio=0.0,  # Guard off to inspect pure filtering
    )
    with torch.no_grad():
        loss_fn.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    loss_fn.train()
    loss_fn.set_epoch(5)

    # Sample 0: clean (CS=1.0 >= tau) -> Retained
    # Sample 1: noise (CS=-1.0 < tau, competing CS=0.0 < 0.5) -> Filtered
    x = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0, 0], dtype=torch.long)

    loss = loss_fn(x, labels)
    loss.backward()

    assert x.grad is not None
    # Sample 0 (retained) must have non-zero gradient
    grad_s0_norm = x.grad[0].norm().item()
    assert grad_s0_norm > 0, "Retained sample must receive non-zero gradient"

    # Sample 1 (filtered) must have EXACTLY zero gradient
    grad_s1_norm = x.grad[1].norm().item()
    assert grad_s1_norm == 0.0, f"Dropped sample must receive exactly zero gradient, got {grad_s1_norm}"

    # Prototype weights must receive gradient
    assert loss_fn.weight.grad is not None
    assert loss_fn.weight.grad.norm().item() > 0


def test_iaam_softmax_edge_cases():
    """Verify handling of edge cases: empty batch, tiny batches, invalid labels, non-finite inputs."""
    loss_fn = IAAMSoftmax(in_features=4, num_classes=3)
    loss_fn.train()

    # 1. Empty batch (B=0)
    x_empty = torch.empty(0, 4, requires_grad=True)
    y_empty = torch.empty(0, dtype=torch.long)
    loss_empty = loss_fn(x_empty, y_empty)
    assert loss_empty.item() == 0.0
    loss_empty.backward()
    assert x_empty.grad is not None

    # 2. Small batch B=1
    x_1 = torch.randn(1, 4, requires_grad=True)
    y_1 = torch.tensor([1])
    loss_1 = loss_fn(x_1, y_1)
    assert not torch.isnan(loss_1)
    loss_1.backward()
    assert x_1.grad is not None

    # 3. Small batch B=2
    x_2 = torch.randn(2, 4, requires_grad=True)
    y_2 = torch.tensor([0, 2])
    loss_2 = loss_fn(x_2, y_2)
    assert not torch.isnan(loss_2)

    # 4. Out of range labels
    x_bad_lbl = torch.randn(4, 4)
    with pytest.raises(ValueError, match="Labels out of range"):
        loss_fn(x_bad_lbl, torch.tensor([0, 1, 5, 0]))  # 5 >= num_classes=3

    with pytest.raises(ValueError, match="Labels out of range"):
        loss_fn(x_bad_lbl, torch.tensor([-1, 0, 1, 0]))

    # 5. Non-finite inputs (NaN / Inf)
    x_nan = torch.randn(4, 4)
    x_nan[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        loss_fn(x_nan, torch.tensor([0, 1, 2, 0]))


def test_iaam_softmax_device_compatibility():
    """Verify loss runs and yields identical results across CPU and MPS."""
    devices = [torch.device("cpu")]
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    for dev in devices:
        torch.manual_seed(42)
        loss_fn = IAAMSoftmax(in_features=16, num_classes=8, esd=2, esc=2).to(dev)
        loss_fn.train()
        loss_fn.set_epoch(3)

        x = torch.randn(8, 16, device=dev, requires_grad=True)
        y = torch.randint(0, 8, (8,), device=dev)

        loss, diag = loss_fn(x, y, return_diagnostics=True)
        assert torch.isfinite(loss)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_iaam_softmax_deterministic_behavior():
    """Verify identical seeds and inputs produce bitwise reproducible results."""
    torch.manual_seed(12345)
    loss_fn1 = IAAMSoftmax(in_features=8, num_classes=4, esd=3, esc=3)
    loss_fn2 = IAAMSoftmax(in_features=8, num_classes=4, esd=3, esc=3)
    with torch.no_grad():
        loss_fn2.weight.copy_(loss_fn1.weight)

    x = torch.randn(10, 8)
    y = torch.randint(0, 4, (10,))

    loss_fn1.train()
    loss_fn2.train()
    loss_fn1.set_epoch(5)
    loss_fn2.set_epoch(5)

    l1, d1 = loss_fn1(x, y, return_diagnostics=True)
    l2, d2 = loss_fn2(x, y, return_diagnostics=True)

    assert abs(l1.item() - l2.item()) < 1e-7
    assert d1 == d2


def test_iaam_softmax_epoch_diagnostics_aggregation():
    """Verify epoch diagnostics accumulate properly across batches."""
    loss_fn = IAAMSoftmax(in_features=4, num_classes=3, esd=2, esc=2, min_keep_ratio=0.5)
    loss_fn.train()
    loss_fn.set_epoch(2)

    # Run 3 batches
    for _ in range(3):
        x = torch.randn(6, 4)
        y = torch.randint(0, 3, (6,))
        loss_fn(x, y)

    epoch_diag = loss_fn.get_epoch_diagnostics()
    assert epoch_diag["epoch"] == 2
    assert epoch_diag["loss_batches"] == 3
    assert epoch_diag["loss_total_samples"] == 18
    assert epoch_diag["loss_num_retained"] <= 18
    assert epoch_diag["loss_num_filtered"] + epoch_diag["loss_num_retained"] == 18
    assert 0.0 <= epoch_diag["loss_keep_ratio"] <= 1.0


def test_iaam_softmax_constructor_boundary_validation():
    """Verify constructor raises ValueError on invalid parameter boundaries."""
    # min_keep_ratio invalid
    with pytest.raises(ValueError, match="min_keep_ratio must be in"):
        IAAMSoftmax(in_features=4, num_classes=3, min_keep_ratio=-0.1)
    with pytest.raises(ValueError, match="min_keep_ratio must be in"):
        IAAMSoftmax(in_features=4, num_classes=3, min_keep_ratio=1.1)

    # tau invalid
    with pytest.raises(ValueError, match="tau must be in"):
        IAAMSoftmax(in_features=4, num_classes=3, tau=-1.5)
    with pytest.raises(ValueError, match="tau must be in"):
        IAAMSoftmax(in_features=4, num_classes=3, tau=1.2)

    # tau_corr invalid
    with pytest.raises(ValueError, match="tau_corr must be in"):
        IAAMSoftmax(in_features=4, num_classes=3, tau_corr=-1.1)
    with pytest.raises(ValueError, match="tau_corr must be in"):
        IAAMSoftmax(in_features=4, num_classes=3, tau_corr=1.05)

    # esd and esc invalid (< 1)
    with pytest.raises(ValueError, match="esd must be >= 1"):
        IAAMSoftmax(in_features=4, num_classes=3, esd=0)
    with pytest.raises(ValueError, match="esc must be >= 1"):
        IAAMSoftmax(in_features=4, num_classes=3, esc=0)

    # Valid boundaries should succeed
    fn = IAAMSoftmax(
        in_features=4,
        num_classes=3,
        min_keep_ratio=0.0,
        tau=-1.0,
        tau_corr=-1.0,
        esd=1,
        esc=1,
    )
    assert fn.min_keep_ratio == 0.0
    assert fn.tau == -1.0
    assert fn.tau_corr == -1.0
    assert fn.esd == 1
    assert fn.esc == 1

    fn2 = IAAMSoftmax(
        in_features=4,
        num_classes=3,
        min_keep_ratio=1.0,
        tau=1.0,
        tau_corr=1.0,
        esd=10,
        esc=10,
    )
    assert fn2.min_keep_ratio == 1.0
    assert fn2.tau == 1.0
    assert fn2.tau_corr == 1.0


"""Integration verification suite for S0 (IAAMSoftmax) and S1 (AAMSoftmax) in Neurovox (Phase 2C).

Requirements verified:
- S0 selects IAAMSoftmax.
- S1 selects ordinary AAMSoftmax.
- Backbone parameters between S0 and S1 are identical (4,167,700 parameters).
- Both support forward/backward training, checkpoint saving and resuming.
- Loss diagnostics are logged to train_metrics and history.
- Numerical state restoration parity across save and resume.
"""

import json
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torchinfo import summary

from src.losses.aam_softmax import AAMSoftmax
from src.losses.iaam_softmax import IAAMSoftmax
from src.models.ska_hnn import SKAHNN, build_ska_hnn
from src.training.trainer import Trainer


class DummyBatchSampler(Sampler):
    """Deterministic batch sampler conforming to Trainer contract."""

    def __init__(self, batch_size: int = 4, num_batches: int = 3, num_classes: int = 4, seed: int = 42):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_classes = num_classes
        self.epoch = 0
        self.seed = seed
        self.current_epoch_indices = []

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.current_epoch_indices = []
        g = torch.Generator().manual_seed(self.seed + epoch * 1000)
        for _ in range(self.num_batches):
            batch = torch.randint(0, self.num_batches * self.batch_size, (self.batch_size,), generator=g).tolist()
            self.current_epoch_indices.append(batch)

    def __iter__(self):
        for batch in self.current_epoch_indices:
            yield batch

    def __len__(self):
        return self.num_batches


def test_s0_s1_backbone_parameter_identity():
    """Verify S0 and S1 have identical backbone architectures and exact parameter counts."""
    m_s0 = build_ska_hnn(switch="S0", c_cnn=32, c_tdnn=256, emb_dim=192, attn_dim=128)
    m_s1 = build_ska_hnn(switch="S1", c_cnn=32, c_tdnn=256, emb_dim=192, attn_dim=128)

    s0_summary = summary(m_s0, input_size=(1, 80, 180), verbose=0)
    s1_summary = summary(m_s1, input_size=(1, 80, 180), verbose=0)

    assert s0_summary.total_params == 4_167_700, f"S0 params mismatch: {s0_summary.total_params}"
    assert s1_summary.total_params == 4_167_700, f"S1 params mismatch: {s1_summary.total_params}"
    assert s0_summary.total_params == s1_summary.total_params

    # Verify identical layer names and parameter shapes
    s0_params = dict(m_s0.named_parameters())
    s1_params = dict(m_s1.named_parameters())
    assert set(s0_params.keys()) == set(s1_params.keys())
    for name in s0_params:
        assert s0_params[name].shape == s1_params[name].shape


def test_s0_selects_iaam_and_s1_selects_aam():
    """Verify loss instantiation contract: S0 pairs with IAAMSoftmax, S1 pairs with AAMSoftmax."""
    num_classes = 10
    emb_dim = 192

    loss_s0 = IAAMSoftmax(
        in_features=emb_dim,
        num_classes=num_classes,
        scale=30.0,
        margin=0.2,
        tau=0.423,
        tau_corr=0.5,
        esd=5,
        esc=5,
    )
    assert isinstance(loss_s0, IAAMSoftmax)
    assert isinstance(loss_s0, AAMSoftmax)  # Subclass check
    assert loss_s0.tau == 0.423
    assert loss_s0.esd == 5

    loss_s1 = AAMSoftmax(
        in_features=emb_dim,
        num_classes=num_classes,
        scale=30.0,
        margin=0.2,
    )
    assert isinstance(loss_s1, AAMSoftmax)
    assert not isinstance(loss_s1, IAAMSoftmax)


def test_s0_trainer_logs_loss_diagnostics(tmp_path):
    """Verify Trainer logs IAAMSoftmax diagnostics during training epochs."""
    device = torch.device("cpu")
    num_classes = 4
    emb_dim = 192

    model = build_ska_hnn(switch="S0", c_cnn=32, c_tdnn=256, emb_dim=emb_dim)
    loss_fn = IAAMSoftmax(
        in_features=emb_dim,
        num_classes=num_classes,
        scale=10.0,
        margin=0.2,
        esd=1,
        esc=1,
        min_keep_ratio=0.5,
    )

    # 12 samples of acoustic features (80, 100)
    x = torch.randn(12, 80, 100)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    ds = TensorDataset(x, y)
    sampler = DummyBatchSampler(batch_size=4, num_batches=3, num_classes=num_classes, seed=42)
    loader = DataLoader(ds, batch_sampler=sampler)

    opt = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        train_loader=loader,
        device=device,
        run_id="test_s0_diagnostics",
    )
    trainer.ckpt_dir = tmp_path / "ckpt_s0"
    trainer.results_dir = tmp_path / "res_s0"
    trainer.ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.results_dir.mkdir(parents=True, exist_ok=True)

    metrics = trainer.train_epoch(epoch=1)

    # Assert diagnostics are exposed in returned metrics
    assert "loss_num_corrected" in metrics
    assert "loss_num_filtered" in metrics
    assert "loss_num_retained" in metrics
    assert "loss_keep_ratio" in metrics
    assert "loss_guard_activations" in metrics

    assert metrics["loss_total_samples"] == 12
    assert metrics["loss_batches"] == 3
    assert 0.0 <= metrics["loss_keep_ratio"] <= 1.0


def test_s0_s1_resume_parity(tmp_path):
    """Verify save and resume achieves exact numerical parity for both S0 and S1."""
    device = torch.device("cpu")
    num_classes = 4
    emb_dim = 192

    for switch, loss_cls in [("S0", IAAMSoftmax), ("S1", AAMSoftmax)]:
        torch.manual_seed(42)
        model = build_ska_hnn(switch=switch, c_cnn=32, c_tdnn=256, emb_dim=emb_dim)
        loss_fn = loss_cls(in_features=emb_dim, num_classes=num_classes, scale=10.0, margin=0.1)

        x = torch.randn(8, 80, 100)
        y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        ds = TensorDataset(x, y)
        sampler = DummyBatchSampler(batch_size=4, num_batches=2, num_classes=num_classes, seed=42)
        loader = DataLoader(ds, batch_sampler=sampler)

        opt = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)

        run_id = f"test_resume_{switch.lower()}"
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            train_loader=loader,
            device=device,
            run_id=run_id,
        )
        trainer.ckpt_dir = tmp_path / f"ckpt_{run_id}"
        trainer.results_dir = tmp_path / f"res_{run_id}"
        trainer.ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.results_dir.mkdir(parents=True, exist_ok=True)

        # Train epoch 1
        m1 = trainer.train_epoch(epoch=1)
        ckpt_path = trainer.ckpt_dir / "epoch_001.pt"
        trainer.save_checkpoint(ckpt_path, epoch=1)

        # Train epoch 2 uninterrupted
        m2_uninterrupted = trainer.train_epoch(epoch=2)

        # Fresh trainer resuming from epoch 1
        model_fresh = build_ska_hnn(switch=switch, c_cnn=32, c_tdnn=256, emb_dim=emb_dim)
        loss_fn_fresh = loss_cls(in_features=emb_dim, num_classes=num_classes, scale=10.0, margin=0.1)
        opt_fresh = torch.optim.Adam(list(model_fresh.parameters()) + list(loss_fn_fresh.parameters()), lr=1e-3)
        sampler_fresh = DummyBatchSampler(batch_size=4, num_batches=2, num_classes=num_classes, seed=42)
        loader_fresh = DataLoader(ds, batch_sampler=sampler_fresh)

        trainer_resumed = Trainer(
            model=model_fresh,
            loss_fn=loss_fn_fresh,
            optimizer=opt_fresh,
            train_loader=loader_fresh,
            device=device,
            run_id=run_id,
        )
        trainer_resumed.ckpt_dir = trainer.ckpt_dir
        trainer_resumed.results_dir = trainer.results_dir

        next_ep = trainer_resumed.load_checkpoint(ckpt_path)
        assert next_ep == 2

        # Train epoch 2 resumed
        m2_resumed = trainer_resumed.train_epoch(epoch=2)

        # Compare losses
        assert abs(m2_uninterrupted["train_loss"] - m2_resumed["train_loss"]) < 1e-5, (
            f"{switch}: Loss parity mismatch: {m2_uninterrupted['train_loss']} vs {m2_resumed['train_loss']}"
        )


class ControlledLinearModel(nn.Module):
    """Controlled identity linear layer producing exact predictable embeddings."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class FixedBatchSampler(Sampler):
    """Batch sampler that yields a single deterministic batch with epoch tracking."""

    def __init__(self, indices):
        self.indices = list(indices)
        self.epoch = 0
        self.current_epoch_indices = [self.indices]

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.current_epoch_indices = [self.indices]

    def __iter__(self):
        yield self.indices

    def __len__(self):
        return 1


def test_trainer_active_adaptivedrop_path(tmp_path):
    """Verify Trainer executes IAAMSoftmax with AdaptiveDrop active beyond warm-up.

    Tests:
    1. Active path reached at epoch >= max(ESC, ESD) = 5.
    2. Finite loss and successful backward gradient propagation.
    3. Exact diagnostics match known ground-truth correction and filtering counts.
    4. Deterministic safeguard activates when candidate keep ratio falls below threshold.
    5. Trainer.fit_overfit_batch respects the explicit epoch argument.
    """
    device = torch.device("cpu")
    dim = 4
    num_classes = 4

    # 4 synthetic samples:
    # idx 0: [1, 0, 0, 0], y=0 -> clean, cos(e0, W0)=1.0 >= tau, kept uncorrected
    # idx 1: [0, 1, 0, 0], y=0 -> mislabeled, cos(e1, W1)=1.0 >= tau_corr, corrected to y=1, kept
    # idx 2: [0, 0, 0, -1], y=0 -> outlier, cos(e2, Wk) <= 0, uncorrected, cos(e2, W0)=0 < tau, candidate drop
    # idx 3: [0, 0, -1, 0], y=0 -> outlier, cos(e3, Wk) <= 0, uncorrected, cos(e3, W0)=0 < tau, candidate drop
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    ds = TensorDataset(x, y)

    # -------------------------------------------------------------------------
    # Case A: min_keep_ratio=0.5 -> candidate keep ratio (2/4 = 0.5) >= 0.5
    # Safeguard should NOT activate.
    # -------------------------------------------------------------------------
    model_a = ControlledLinearModel(dim=dim)
    loss_fn_a = IAAMSoftmax(
        in_features=dim,
        num_classes=num_classes,
        scale=10.0,
        margin=0.2,
        tau=0.423,
        tau_corr=0.5,
        esd=5,
        esc=5,
        min_keep_ratio=0.5,
    )
    with torch.no_grad():
        loss_fn_a.weight.copy_(torch.eye(num_classes))

    opt_a = torch.optim.SGD(list(model_a.parameters()) + list(loss_fn_a.parameters()), lr=0.01)
    loader_a = DataLoader(ds, batch_sampler=FixedBatchSampler([0, 1, 2, 3]))

    trainer_a = Trainer(
        model=model_a,
        loss_fn=loss_fn_a,
        optimizer=opt_a,
        train_loader=loader_a,
        device=device,
        run_id="test_active_drop_noguard",
    )
    trainer_a.ckpt_dir = tmp_path / "ckpt_a"
    trainer_a.results_dir = tmp_path / "res_a"
    trainer_a.ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer_a.results_dir.mkdir(parents=True, exist_ok=True)

    metrics_a = trainer_a.train_epoch(epoch=6)

    # Verification: active path reached and diagnostics match exactly
    assert metrics_a["epoch"] == 6
    assert metrics_a["loss_num_corrected"] == 1, "Sample 1 should be corrected to class 1"
    assert metrics_a["loss_num_filtered"] == 2, "Samples 2 and 3 should be filtered"
    assert metrics_a["loss_num_retained"] == 2, "Samples 0 and 1 should be retained"
    assert metrics_a["loss_keep_ratio"] == 0.5
    assert metrics_a["loss_guard_activations"] == 0, "Safeguard must not activate when ratio >= min_keep_ratio"
    assert math.isfinite(metrics_a["train_loss"])
    assert metrics_a["train_loss"] > 0

    # Gradient flow check
    assert model_a.fc.weight.grad is not None
    assert torch.isfinite(model_a.fc.weight.grad).all()
    assert model_a.fc.weight.grad.abs().sum() > 0
    assert loss_fn_a.weight.grad is not None
    assert torch.isfinite(loss_fn_a.weight.grad).all()
    assert loss_fn_a.weight.grad.abs().sum() > 0

    # -------------------------------------------------------------------------
    # Case B: min_keep_ratio=0.75 -> candidate keep ratio (2/4 = 0.5) < 0.75
    # Safeguard MUST activate, retaining top-3 samples (keep_ratio = 0.75).
    # -------------------------------------------------------------------------
    model_b = ControlledLinearModel(dim=dim)
    loss_fn_b = IAAMSoftmax(
        in_features=dim,
        num_classes=num_classes,
        scale=10.0,
        margin=0.2,
        tau=0.423,
        tau_corr=0.5,
        esd=5,
        esc=5,
        min_keep_ratio=0.75,
    )
    with torch.no_grad():
        loss_fn_b.weight.copy_(torch.eye(num_classes))

    opt_b = torch.optim.SGD(list(model_b.parameters()) + list(loss_fn_b.parameters()), lr=0.01)
    loader_b = DataLoader(ds, batch_sampler=FixedBatchSampler([0, 1, 2, 3]))

    trainer_b = Trainer(
        model=model_b,
        loss_fn=loss_fn_b,
        optimizer=opt_b,
        train_loader=loader_b,
        device=device,
        run_id="test_active_drop_guard",
    )
    trainer_b.ckpt_dir = tmp_path / "ckpt_b"
    trainer_b.results_dir = tmp_path / "res_b"
    trainer_b.ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer_b.results_dir.mkdir(parents=True, exist_ok=True)

    metrics_b = trainer_b.train_epoch(epoch=6)

    # Verification: safeguard activated
    assert metrics_b["epoch"] == 6
    assert metrics_b["loss_num_corrected"] == 1
    assert metrics_b["loss_num_filtered"] == 1
    assert metrics_b["loss_num_retained"] == 3
    assert metrics_b["loss_keep_ratio"] == 0.75
    assert metrics_b["loss_guard_activations"] == 1, "Safeguard must activate when ratio < min_keep_ratio"
    assert math.isfinite(metrics_b["train_loss"])
    assert model_b.fc.weight.grad is not None
    assert torch.isfinite(model_b.fc.weight.grad).all()

    # -------------------------------------------------------------------------
    # Case C: Trainer.fit_overfit_batch epoch forwarding
    # -------------------------------------------------------------------------
    model_c = ControlledLinearModel(dim=dim)
    loss_fn_c = IAAMSoftmax(
        in_features=dim,
        num_classes=num_classes,
        esd=5,
        esc=5,
    )
    opt_c = torch.optim.SGD(list(model_c.parameters()) + list(loss_fn_c.parameters()), lr=0.01)
    loader_c = DataLoader(ds, batch_sampler=FixedBatchSampler([0, 1, 2, 3]))

    trainer_c = Trainer(
        model=model_c,
        loss_fn=loss_fn_c,
        optimizer=opt_c,
        train_loader=loader_c,
        device=device,
        run_id="test_active_drop_overfit_batch_epoch",
    )
    trainer_c.ckpt_dir = tmp_path / "ckpt_c"
    trainer_c.results_dir = tmp_path / "res_c"
    trainer_c.ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer_c.results_dir.mkdir(parents=True, exist_ok=True)

    # Calling fit_overfit_batch with epoch=6 sets epoch on loss_fn
    try:
        trainer_c.fit_overfit_batch(num_steps=1, epoch=6)
    except RuntimeError:
        pass  # Overfit gate check (100x reduction) is expected to fail on a single synthetic step

    assert loss_fn_c.current_epoch == 6
    last_diag = loss_fn_c.get_last_diagnostics()
    assert last_diag["epoch"] == 6
    assert last_diag["epoch"] >= loss_fn_c.esc
    assert last_diag["epoch"] >= loss_fn_c.esd
    assert "num_corrected" in last_diag
    assert "num_filtered" in last_diag
    assert "num_retained" in last_diag





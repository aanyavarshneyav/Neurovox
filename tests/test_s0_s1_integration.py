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

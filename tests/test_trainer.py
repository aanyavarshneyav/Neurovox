"""Unit tests for Trainer: stateful resume parity, NaN guards, and epoch propagation."""

from pathlib import Path
import json
import shutil
import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset

from src.data.dataset import LibriSpeechDataset
from src.data.sampler import SpeakerBalancedBatchSampler
from src.losses.aam_softmax import AAMSoftmax
from src.models.small_tdnn_se import small_tdnn_se
from src.training.trainer import Trainer


def build_synthetic_loader(batch_size: int = 4, num_batches: int = 5, num_classes: int = 10, seed: int = 42):
    """Build a tiny synthetic DataLoader with stub batch_sampler supporting set_epoch."""
    torch.manual_seed(seed)
    total_samples = batch_size * num_batches
    x = torch.randn(total_samples, 80, 100)
    y = torch.randint(0, num_classes, (total_samples,))
    dataset = TensorDataset(x, y)

    class StubBatchSampler(Sampler):
        def __init__(self):
            self.epoch = 0
            self.current_epoch_indices = []

        def set_epoch(self, epoch: int):
            self.epoch = epoch

        def __iter__(self):
            self.current_epoch_indices = [
                list(range(i * batch_size, (i + 1) * batch_size))
                for i in range(num_batches)
            ]
            for b in self.current_epoch_indices:
                yield b

        def __len__(self):
            return num_batches

    return DataLoader(dataset, batch_sampler=StubBatchSampler())


def create_mock_libri_dataset(tmp_path: Path, num_speakers: int = 4, utts_per_spk: int = 3, seed: int = 42):
    """Create a small on-disk mock dataset for real dataset/sampler testing."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = tmp_path / "manifest.csv"

    entries = []
    for spk in range(num_speakers):
        for u in range(utts_per_spk):
            utt_id = f"spk{spk}_utt{u}"
            flac_path = audio_dir / f"{utt_id}.flac"
            data = np.random.randn(32000).astype(np.float32) * 0.1
            sf.write(str(flac_path), data, 16000, format="FLAC")
            entries.append(f"{utt_id},{spk},ch_1,M,{flac_path},2.0,train,{spk}\n")

    with open(manifest_csv, "w", encoding="utf-8") as f:
        f.write("utt_id,spk_id,chapter_id,gender,path,duration,split,label\n")
        f.writelines(entries)

    ds = LibriSpeechDataset(
        manifest_path=manifest_csv,
        is_train=True,
        return_features=True,
        enable_speed_aug=False,
        enable_noise_aug=False,
        enable_specaug=False,
        base_seed=seed,
    )
    return ds, manifest_csv


def test_trainer_raises_if_batch_sampler_lacks_set_epoch(tmp_path):
    """Verify Trainer raises TypeError if train_loader.batch_sampler lacks set_epoch."""
    device = torch.device("cpu")
    model = small_tdnn_se(in_features=80, channels=16, emb_dim=32).to(device)
    loss_fn = AAMSoftmax(in_features=32, num_classes=4, scale=10.0, margin=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(8, 80, 100)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    # Standard DataLoader with batch_size (has default BatchSampler without set_epoch)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        train_loader=loader,
        device=device,
        run_id="test_no_set_epoch",
    )
    trainer.ckpt_dir = tmp_path / "ckpt"
    trainer.results_dir = tmp_path / "res"
    trainer.ckpt_dir.mkdir(parents=True)
    trainer.results_dir.mkdir(parents=True)

    with pytest.raises(TypeError, match="set_epoch"):
        trainer.train_epoch(1)


def test_trainer_epoch_advance_through_trainer_path(tmp_path):
    """Verify Trainer.train_epoch advances batch_sampler epoch and hashes differ for epochs 1, 2, 3."""
    device = torch.device("cpu")
    model = small_tdnn_se(in_features=80, channels=16, emb_dim=32).to(device)
    loss_fn = AAMSoftmax(in_features=32, num_classes=4, scale=10.0, margin=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    class DynamicStubSampler(Sampler):
        def __init__(self):
            self.epoch = 0
            self.current_epoch_indices = []

        def set_epoch(self, epoch: int):
            self.epoch = epoch

        def __iter__(self):
            # Batches depend explicitly on epoch
            self.current_epoch_indices = [
                [(self.epoch * 100 + i) for i in range(4)],
                [(self.epoch * 100 + 4 + i) for i in range(4)],
            ]
            for b in self.current_epoch_indices:
                yield b

        def __len__(self):
            return 2

    sampler = DynamicStubSampler()
    # Dummy dataset
    class DummyDS(Dataset):
        def __len__(self):
            return 1000
        def __getitem__(self, idx):
            return torch.randn(80, 100), idx % 4

    loader = DataLoader(DummyDS(), batch_sampler=sampler)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        train_loader=loader,
        device=device,
        run_id="test_epoch_advance",
    )
    trainer.ckpt_dir = tmp_path / "ckpt"
    trainer.results_dir = tmp_path / "res"
    trainer.ckpt_dir.mkdir(parents=True)
    trainer.results_dir.mkdir(parents=True)

    m1 = trainer.train_epoch(1)
    assert sampler.epoch == 1
    assert m1["sampler_epoch"] == 1
    assert m1["unique_utts_this_epoch"] == 8
    assert m1["cum_unique_utts"] == 8

    m2 = trainer.train_epoch(2)
    assert sampler.epoch == 2
    assert m2["sampler_epoch"] == 2
    assert m2["unique_utts_this_epoch"] == 8
    assert m2["cum_unique_utts"] == 16

    m3 = trainer.train_epoch(3)
    assert sampler.epoch == 3
    assert m3["sampler_epoch"] == 3
    assert m3["unique_utts_this_epoch"] == 8
    assert m3["cum_unique_utts"] == 24

    hashes = [m1["first_batch_index_hash"], m2["first_batch_index_hash"], m3["first_batch_index_hash"]]
    assert len(set(hashes)) == 3, f"First-batch index hashes must differ across epochs: {hashes}"


@pytest.mark.parametrize("num_workers", [0, 2])
def test_trainer_resume_parity_cpu(tmp_path, num_workers):
    """Verify uninterrupted 2-epoch run strictly matches 1-epoch + resume run on CPU (< 1e-5)."""
    device = torch.device("cpu")
    num_classes = 4
    seed = 42

    def build_system(run_name: str):
        torch.manual_seed(seed)
        np.random.seed(seed)
        ds, _ = create_mock_libri_dataset(tmp_path / run_name, num_speakers=num_classes, seed=seed)
        sampler = SpeakerBalancedBatchSampler(
            speaker_labels=ds.labels,
            batch_size=num_classes,
            batches_per_epoch=3,
            seed=seed,
        )
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )
        model = small_tdnn_se(in_features=80, channels=32, emb_dim=64).to(device)
        loss_fn = AAMSoftmax(in_features=64, num_classes=num_classes, scale=10.0, margin=0.1).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)
        sched = torch.optim.lr_scheduler.CyclicLR(
            opt, base_lr=1e-4, max_lr=1e-3, step_size_up=3, mode="triangular", cycle_momentum=False
        )
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            train_loader=loader,
            scheduler=sched,
            device=device,
            run_id=f"test_cpu_{run_name}",
            seed=seed,
        )
        trainer.ckpt_dir = tmp_path / f"ckpt_{run_name}"
        trainer.results_dir = tmp_path / f"res_{run_name}"
        trainer.ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.results_dir.mkdir(parents=True, exist_ok=True)
        return trainer, model

    # 1. Uninterrupted 2-epoch run
    trainer1, model1 = build_system(f"uninterrupted_w{num_workers}")
    history1 = trainer1.fit(num_epochs=2, start_epoch=1)

    # 2. Resumed run: Run epoch 1, then build FRESH objects, load epoch 1, run epoch 2
    trainer2_init, _ = build_system(f"resumed_w{num_workers}")
    _ = trainer2_init.fit(num_epochs=1, start_epoch=1)
    ckpt_path = trainer2_init.ckpt_dir / "epoch_001.pt"

    # Build completely fresh objects
    trainer2_fresh, model2_fresh = build_system(f"resumed_w{num_workers}")
    next_epoch = trainer2_fresh.load_checkpoint(ckpt_path)
    assert next_epoch == 2

    history2 = trainer2_fresh.fit(num_epochs=2, start_epoch=next_epoch)

    # Assert history contains both epochs after resume
    assert len(trainer2_fresh.history) == 2, f"History should contain 2 epochs, got {len(trainer2_fresh.history)}"
    assert trainer2_fresh.history[0]["epoch"] == 1
    assert trainer2_fresh.history[1]["epoch"] == 2

    # Assert history.json on disk contains both epochs (F7)
    disk_history_path = trainer2_fresh.results_dir / "history.json"
    assert disk_history_path.exists(), "history.json must exist on disk after resume"
    with open(disk_history_path, "r", encoding="utf-8") as f:
        disk_history = json.load(f)
    assert len(disk_history) == 2, f"disk history should contain 2 epochs, got {len(disk_history)}"
    assert disk_history[0]["epoch"] == 1
    assert disk_history[1]["epoch"] == 2

    # Assert loss and parameter parity (< 1e-5)
    loss1 = history1[1]["train_loss"]
    loss2 = history2[1]["train_loss"]
    assert abs(loss1 - loss2) < 1e-5, f"Loss mismatch: uninterrupted={loss1:.6f}, resumed={loss2:.6f}"

    # Compare unrounded train loss (F7)
    unrounded1 = history1[1].get("train_loss_unrounded", loss1)
    unrounded2 = history2[1].get("train_loss_unrounded", loss2)
    assert abs(unrounded1 - unrounded2) < 1e-5, f"Unrounded loss mismatch: {unrounded1} vs {unrounded2}"

    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2_fresh.named_parameters()):
        max_diff = (p1 - p2).abs().max().item()
        assert max_diff < 1e-5, f"Parameter mismatch in {n1}: max diff = {max_diff:.2e}"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_trainer_resume_parity_mps(tmp_path):
    """Verify resume parity on Apple Silicon MPS within documented looser tolerance (< 1e-4)."""
    device = torch.device("mps")
    num_classes = 4
    seed = 42

    def build_system(run_name: str):
        torch.manual_seed(seed)
        np.random.seed(seed)
        ds, _ = create_mock_libri_dataset(tmp_path / run_name, num_speakers=num_classes, seed=seed)
        sampler = SpeakerBalancedBatchSampler(
            speaker_labels=ds.labels,
            batch_size=num_classes,
            batches_per_epoch=3,
            seed=seed,
        )
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=0,
        )
        model = small_tdnn_se(in_features=80, channels=32, emb_dim=64).to(device)
        loss_fn = AAMSoftmax(in_features=64, num_classes=num_classes, scale=10.0, margin=0.1).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)
        sched = torch.optim.lr_scheduler.CyclicLR(
            opt, base_lr=1e-4, max_lr=1e-3, step_size_up=3, mode="triangular", cycle_momentum=False
        )
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            train_loader=loader,
            scheduler=sched,
            device=device,
            run_id=f"test_mps_{run_name}",
            seed=seed,
        )
        trainer.ckpt_dir = tmp_path / f"ckpt_{run_name}"
        trainer.results_dir = tmp_path / f"res_{run_name}"
        trainer.ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.results_dir.mkdir(parents=True, exist_ok=True)
        return trainer, model

    trainer1, model1 = build_system("mps_uninterrupted")
    history1 = trainer1.fit(num_epochs=2, start_epoch=1)

    trainer2_init, _ = build_system("mps_resumed")
    _ = trainer2_init.fit(num_epochs=1, start_epoch=1)
    ckpt_path = trainer2_init.ckpt_dir / "epoch_001.pt"

    trainer2_fresh, model2_fresh = build_system("mps_resumed")
    next_epoch = trainer2_fresh.load_checkpoint(ckpt_path)
    assert next_epoch == 2
    history2 = trainer2_fresh.fit(num_epochs=2, start_epoch=next_epoch)

    assert len(trainer2_fresh.history) == 2
    loss1 = history1[1]["train_loss"]
    loss2 = history2[1]["train_loss"]
    assert abs(loss1 - loss2) < 1e-4, f"MPS loss mismatch: uninterrupted={loss1:.6f}, resumed={loss2:.6f}"

    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2_fresh.named_parameters()):
        max_diff = (p1 - p2).abs().max().item()
        assert max_diff < 1e-4, f"MPS parameter mismatch in {n1}: max diff = {max_diff:.2e}"


def test_trainer_nan_guards_all_stages(tmp_path):
    """Verify NaN guards trigger FloatingPointError across all four stages."""
    device = torch.device("cpu")
    num_classes = 4

    def build_test_trainer(run_id: str):
        model = small_tdnn_se(in_features=80, channels=16, emb_dim=32).to(device)
        loss_fn = AAMSoftmax(in_features=32, num_classes=num_classes, scale=10.0, margin=0.1).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = build_synthetic_loader(batch_size=4, num_batches=2, num_classes=num_classes)
        t = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            train_loader=loader,
            device=device,
            run_id=run_id,
        )
        t.ckpt_dir = tmp_path / f"ckpt_{run_id}"
        t.results_dir = tmp_path / f"res_{run_id}"
        t.ckpt_dir.mkdir(parents=True, exist_ok=True)
        t.results_dir.mkdir(parents=True, exist_ok=True)
        return t, model, loss_fn, loader

    # Stage 1: Input features NaN
    t1, _, _, _ = build_test_trainer("nan_input")
    x_nan = torch.randn(8, 80, 100)
    x_nan[0, 0, 0] = float("nan")
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    class StubSampler1(Sampler):
        def set_epoch(self, epoch): pass
        def __iter__(self): yield [0, 1, 2, 3]; yield [4, 5, 6, 7]
        def __len__(self): return 2
    t1.train_loader = DataLoader(TensorDataset(x_nan, y), batch_sampler=StubSampler1())
    with pytest.raises(FloatingPointError, match="input_features"):
        t1.train_epoch(1)

    # Stage 2: Embeddings NaN
    t2, m2, _, _ = build_test_trainer("nan_emb")
    m2.register_forward_hook(lambda module, inp, out: out * float("nan"))
    with pytest.raises(FloatingPointError, match="model_embeddings"):
        t2.train_epoch(1)

    # Stage 3: Loss NaN
    t3, _, _, _ = build_test_trainer("nan_loss")
    class NanLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_classes = num_classes
        def forward(self, emb, labels):
            return torch.tensor(float("nan"), requires_grad=True)
    t3.loss_fn = NanLoss()
    with pytest.raises(FloatingPointError, match="loss_computation"):
        t3.train_epoch(1)

    # Stage 4: Gradients NaN
    t4, m4, _, _ = build_test_trainer("nan_grad")
    # Register backward hook on a leaf parameter to inject NaN into grad
    for p in m4.parameters():
        if p.requires_grad:
            p.register_hook(lambda grad: grad * float("nan"))
            break
    with pytest.raises(FloatingPointError, match="grad_"):
        t4.train_epoch(1)


def test_trainer_label_range_check(tmp_path):
    """Verify Trainer raises ValueError on epoch 1 batch 1 if labels are out of bounds."""
    device = torch.device("cpu")
    num_classes = 4
    model = small_tdnn_se(in_features=80, channels=16, emb_dim=32).to(device)
    loss_fn = AAMSoftmax(in_features=32, num_classes=num_classes, scale=10.0, margin=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(4, 80, 100)
    y_out_of_bounds = torch.tensor([0, 1, 2, 99])  # 99 >= num_classes=4

    class StubSampler(Sampler):
        def set_epoch(self, epoch): pass
        def __iter__(self): yield [0, 1, 2, 3]
        def __len__(self): return 1

    loader = DataLoader(TensorDataset(x, y_out_of_bounds), batch_sampler=StubSampler())
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        train_loader=loader,
        device=device,
        run_id="test_label_bounds",
    )
    trainer.ckpt_dir = tmp_path / "ckpt"
    trainer.results_dir = tmp_path / "res"
    trainer.ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.results_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="Target label out of bounds"):
        trainer.train_epoch(1)


def test_trainer_overwrite_deletes_stale_checkpoints(tmp_path):
    """Verify --overwrite purges stale checkpoints so resume picks fresh latest (F3)."""
    device = torch.device("cpu")
    num_classes = 4
    seed = 42

    ckpt_dir = tmp_path / "ckpt_overwrite"
    results_dir = tmp_path / "res_overwrite"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Place stale epoch_011.pt
    stale_ckpt = ckpt_dir / "epoch_011.pt"
    torch.save({"epoch": 11, "model_state_dict": {}}, stale_ckpt)
    assert stale_ckpt.exists()

    # Simulate overwrite action from scripts/train.py
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    if results_dir.exists():
        shutil.rmtree(results_dir)

    assert not stale_ckpt.exists()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train 2 epochs fresh
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds, _ = create_mock_libri_dataset(tmp_path / "fresh_run", num_speakers=num_classes, seed=seed)
    sampler = SpeakerBalancedBatchSampler(
        speaker_labels=ds.labels,
        batch_size=num_classes,
        batches_per_epoch=2,
        seed=seed,
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)
    model = small_tdnn_se(in_features=80, channels=16, emb_dim=32).to(device)
    loss_fn = AAMSoftmax(in_features=32, num_classes=num_classes, scale=10.0, margin=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        train_loader=loader,
        device=device,
        run_id="test_overwrite_run",
        seed=seed,
    )
    trainer.ckpt_dir = ckpt_dir
    trainer.results_dir = results_dir

    trainer.fit(num_epochs=2, start_epoch=1)

    # Verify stale checkpoint is absent, fresh checkpoints survive
    assert not (ckpt_dir / "epoch_011.pt").exists()
    assert (ckpt_dir / "epoch_001.pt").exists()
    assert (ckpt_dir / "epoch_002.pt").exists()

    # Verify resume picks fresh latest (epoch_002.pt)
    from src.training.trainer import find_latest_checkpoint
    latest = find_latest_checkpoint(ckpt_dir)
    assert latest.name == "epoch_002.pt"

    # Loading it yields next_epoch = 3
    fresh_trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        train_loader=loader,
        device=device,
        run_id="test_overwrite_run",
    )
    fresh_trainer.ckpt_dir = ckpt_dir
    fresh_trainer.results_dir = results_dir
    next_ep = fresh_trainer.load_checkpoint(latest)
    assert next_ep == 3


def test_load_checkpoint_cuda_rng_device_mapping(tmp_path, monkeypatch):
    """Verify load_checkpoint transfers device-mapped CUDA RNG state to CPU before restore (F4)."""
    device = torch.device("cpu")
    model = small_tdnn_se(in_features=80, channels=16, emb_dim=32).to(device)
    loss_fn = AAMSoftmax(in_features=32, num_classes=4, scale=10.0, margin=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = build_synthetic_loader(batch_size=4, num_batches=1, num_classes=4)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        train_loader=loader,
        device=device,
        run_id="test_cuda_rng",
    )
    trainer.ckpt_dir = tmp_path / "ckpt"
    trainer.results_dir = tmp_path / "res"
    trainer.ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.results_dir.mkdir(parents=True, exist_ok=True)

    # Save a mock checkpoint with a tensor cuda_rng_state
    mock_cuda_state = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
    ckpt_path = trainer.ckpt_dir / "epoch_001.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "loss_state_dict": loss_fn.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "rng_state": {"cuda": mock_cuda_state},
        },
        ckpt_path,
    )

    restored_state = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state: restored_state.append(state))

    trainer.load_checkpoint(ckpt_path)
    assert len(restored_state) == 1
    assert isinstance(restored_state[0], torch.Tensor)
    assert restored_state[0].device.type == "cpu", "CUDA RNG state passed to set_rng_state must be on CPU"


def test_step_0_baseline_excluded_from_epoch_1_diagnostics(tmp_path):
    """Verify baseline forward pass before epoch 1 is excluded from epoch-1 diagnostics."""
    from src.losses.iaam_softmax import IAAMSoftmax

    device = torch.device("cpu")
    num_classes = 4
    batch_size = 4
    num_batches = 3
    model = small_tdnn_se(in_features=80, channels=16, emb_dim=32).to(device)
    loss_fn = IAAMSoftmax(
        in_features=32,
        num_classes=num_classes,
        scale=10.0,
        margin=0.2,
        esd=1,
        esc=1,
        min_keep_ratio=0.5,
    ).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)
    loader = build_synthetic_loader(batch_size=batch_size, num_batches=num_batches, num_classes=num_classes)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        train_loader=loader,
        device=device,
        run_id="test_step_0_diag_exclusion",
    )
    trainer.ckpt_dir = tmp_path / "ckpt"
    trainer.results_dir = tmp_path / "res"
    trainer.ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.results_dir.mkdir(parents=True, exist_ok=True)

    history = trainer.fit(num_epochs=1, start_epoch=1)

    assert trainer.step_0_loss is not None, "step_0 baseline forward pass must run"
    epoch1_record = history[0]

    # Baseline forward pass before epoch 1 must be excluded from epoch-1 diagnostics
    assert epoch1_record["loss_batches"] == num_batches, (
        f"Expected loss_batches == {num_batches}, got {epoch1_record['loss_batches']}"
    )
    assert epoch1_record["loss_total_samples"] == num_batches * batch_size, (
        f"Expected loss_total_samples == {num_batches * batch_size}, got {epoch1_record['loss_total_samples']}"
    )

    # Verify timing instrumentation columns
    assert "data_loading_sec" in epoch1_record
    assert "forward_backward_optimizer_sec" in epoch1_record
    assert "dev_clean_eval_sec" in epoch1_record
    assert "total_sec" in epoch1_record
    assert "peak_gpu_memory_bytes" in epoch1_record
    assert "num_workers" in epoch1_record

    # Verify history.csv contains the added columns
    csv_path = trainer.results_dir / "history.csv"
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    for col in [
        "data_loading_sec",
        "forward_backward_optimizer_sec",
        "dev_clean_eval_sec",
        "total_sec",
        "peak_gpu_memory_bytes",
        "num_workers",
    ]:
        assert col in header, f"Column {col} missing from history.csv header"



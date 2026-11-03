#!/usr/bin/env python3
"""Main training script for Neurovox speaker verification models.

Supports:
- Model instantiation (small_tdnn_se for Phase 2A).
- Margin loss (AAMSoftmax in fp32).
- Speaker-balanced batch sampling with distinct speakers per batch.
- Periodic full-utterance evaluation on dev-clean and train-probe (training-data EER).
- Stateful resume (--resume) with RNG restoration.
- Explicit device and precision selection (device=mps, fp32 default on Mac).
"""

import argparse
import hashlib
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Dict

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import LibriSpeechDataset
from src.data.sampler import SpeakerBalancedBatchSampler
from src.losses.aam_softmax import AAMSoftmax
from src.models import build_ska_hnn_from_config, small_tdnn_se
from src.training.trainer import Trainer, find_latest_checkpoint
from src.utils.paths import get_ckpt_root, get_data_root, get_results_root, resolve_path


class ConfigDict(dict):
    """Dictionary supporting attribute access and nested dict conversion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict):
                self[k] = ConfigDict(v)
            elif isinstance(v, list):
                self[k] = [ConfigDict(item) if isinstance(item, dict) else item for item in v]

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: Path) -> ConfigDict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ConfigDict(data)


def compute_config_hash(config_path: Path) -> str:
    """Compute 8-character SHA-256 hash of configuration file."""
    h = hashlib.sha256(config_path.read_bytes())
    return h.hexdigest()[:8]


def main() -> int:
    parser = argparse.ArgumentParser(description="Neurovox Model Training Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID identifier (defaults to config stem)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint in CKPT_ROOT / run_id",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing checkpoints and results for this run_id (deletes existing directories)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to target ('mps', 'cuda', 'cpu', 'auto')",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32"],
        help="Numerical precision (strictly 'fp32' per ADR-007)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}", file=sys.stderr)
        return 1

    cfg = load_config(config_path)
    config_hash = compute_config_hash(config_path)

    run_id = args.run_id if args.run_id else config_path.stem
    num_epochs = args.epochs if args.epochs is not None else int(cfg.training.get("epochs", 1))
    seed = int(cfg.training.get("seed", 42))

    # Directories
    ckpt_dir = get_ckpt_root() / run_id
    results_dir = get_results_root() / run_id

    # F3: Handle --overwrite by strictly deleting existing directories
    if args.overwrite:
        if ckpt_dir.exists():
            print(f"[OVERWRITE] Deleting existing checkpoint directory: {ckpt_dir}")
            shutil.rmtree(ckpt_dir)
        if results_dir.exists():
            print(f"[OVERWRITE] Deleting existing results directory: {results_dir}")
            shutil.rmtree(results_dir)
    elif not args.resume:
        ckpt_has_files = ckpt_dir.exists() and any(ckpt_dir.iterdir())
        res_has_files = results_dir.exists() and any(results_dir.iterdir())
        if ckpt_has_files or res_has_files:
            raise FileExistsError(
                f"Output directory already contains files for run '{run_id}'. "
                f"Pass --resume to resume or --overwrite to overwrite."
            )

    # Set seeds across all RNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Dataset & Sampler setup
    manifest_path = resolve_path(cfg.data.get("train_manifest", "data/manifests/train-clean-100.csv"))
    batch_size = int(cfg.training.get("batch_size", 64))
    batches_per_epoch = cfg.training.get("batches_per_epoch", None)
    if batches_per_epoch is not None:
        batches_per_epoch = int(batches_per_epoch)

    # Data augmentations
    aug_cfg = cfg.get("augmentation", {})
    enable_speed = bool(aug_cfg.get("speed_perturbation", {}).get("enabled", False))
    enable_noise = bool(aug_cfg.get("synthetic_noise", {}).get("enabled", False))
    enable_babble = bool(aug_cfg.get("synthetic_noise", {}).get("babble_prob", 0.0) > 0.0)
    enable_rir = bool(aug_cfg.get("rirs_noises", {}).get("enabled", False))
    enable_specaug = bool(aug_cfg.get("specaugment", {}).get("enabled", True))

    speed_prob = float(aug_cfg.get("speed_perturbation", {}).get("prob", 0.5))
    noise_prob = float(aug_cfg.get("synthetic_noise", {}).get("gaussian_prob", 0.3))
    babble_prob = float(aug_cfg.get("synthetic_noise", {}).get("babble_prob", 0.4))
    rir_prob = float(aug_cfg.get("rirs_noises", {}).get("prob", 0.5))
    specaug_p = float(aug_cfg.get("specaugment", {}).get("p", 0.5))

    print("=" * 80)
    print("NEUROVOX TRAINING PIPELINE")
    print(f"Run ID:         {run_id}")
    print(f"Config:         {config_path} (hash: {config_hash})")
    print(f"Device:         {device} | Precision: {args.precision}")
    print(f"Seed:           {seed}")
    print(f"Augmentations:  speed={enable_speed} (p={speed_prob}), noise={enable_noise} (p={noise_prob}), babble={enable_babble} (p={babble_prob}), rir={enable_rir} (p={rir_prob}), specaug={enable_specaug} (p={specaug_p})")
    print(f"Epochs:         {num_epochs} | Batch Size: {batch_size}")
    print("=" * 80)

    dataset = LibriSpeechDataset(
        manifest_path=manifest_path,
        target_samples=int(cfg.audio.get("segment_frames", 180)) * 160 + 240,  # 29040 samples
        is_train=True,
        return_features=True,  # Returns 80-dim log-Mel features (80, 180)
        enable_speed_aug=enable_speed,
        enable_noise_aug=enable_noise,
        enable_babble_aug=enable_babble,
        enable_rir_aug=enable_rir,
        enable_specaug=enable_specaug,
        speed_prob=speed_prob,
        noise_prob=noise_prob,
        babble_prob=babble_prob,
        rir_prob=rir_prob,
        specaug_p=specaug_p,
        base_seed=seed,
    )

    # F8: Integrity assertions before training
    labels = [int(entry["label"]) for entry in dataset.entries]
    unique_labels = sorted(set(labels))
    N = len(unique_labels)
    assert unique_labels == list(range(N)), f"Labels are not contiguous 0..{N-1}: {unique_labels[:10]}..."
    num_classes = int(cfg.loss.get("num_classes", 251))
    assert num_classes == N, f"Config num_classes ({num_classes}) != unique label count ({N})"

    # Assert spk_id <-> label is strictly one-to-one
    spk2label = {}
    label2spk = {}
    for entry in dataset.entries:
        spk = entry["spk_id"]
        lbl = int(entry["label"])
        if spk in spk2label and spk2label[spk] != lbl:
            raise AssertionError(f"Inconsistent speaker mapping: {spk} mapped to {spk2label[spk]} and {lbl}")
        if lbl in label2spk and label2spk[lbl] != spk:
            raise AssertionError(f"Inconsistent label mapping: {lbl} mapped to {label2spk[lbl]} and {spk}")
        spk2label[spk] = lbl
        label2spk[lbl] = spk

    speaker_labels = [row["spk_id"] for row in dataset.entries]
    sampler = SpeakerBalancedBatchSampler(
        speaker_labels=speaker_labels,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        seed=seed,
    )

    num_workers = int(cfg.hardware.get("num_workers", 2))
    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=bool(cfg.hardware.get("pin_memory", False)),
    )

    # Model instantiation keyed on cfg.model.name
    model_name = cfg.model.get("name", "small_tdnn_se")
    emb_dim = int(cfg.model.get("emb_dim", 192))
    if model_name == "small_tdnn_se":
        in_features = int(cfg.audio.get("n_mels", 80))
        channels = int(cfg.model.get("channels", 128))
        model = small_tdnn_se(in_features=in_features, channels=channels, emb_dim=emb_dim).to(device)
    elif model_name == "ska_hnn":
        model = build_ska_hnn_from_config(cfg).to(device)
    else:
        raise ValueError(f"Unknown model architecture: '{model_name}'")

    # Loss: AAMSoftmax
    scale = float(cfg.loss.get("scale", 30.0))
    margin = float(cfg.loss.get("margin", 0.2))
    loss_fn = AAMSoftmax(
        in_features=emb_dim,
        num_classes=num_classes,
        scale=scale,
        margin=margin,
    ).to(device)

    # Optimizer
    weight_decay = float(cfg.training.get("weight_decay", 1e-4))
    lr_init = float(cfg.training.get("lr_init", 1e-7))
    lr_max = float(cfg.training.get("lr_max", 1e-3))
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=lr_init,
        weight_decay=weight_decay,
    )

    # Scheduler: CyclicLR or None (for constant LR)
    schedule_type = cfg.training.get("lr_schedule", "cyclic")
    if schedule_type == "cyclic":
        cycle_epochs = int(cfg.training.get("cycle_epochs", 2))
        step_size_up = max(1, (len(train_loader) * cycle_epochs) // 2)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr_init,
            max_lr=lr_max,
            step_size_up=step_size_up,
            mode="triangular",
            cycle_momentum=False,
        )
    else:
        # Constant LR
        optimizer.param_groups[0]["lr"] = lr_max
        scheduler = None

    # Evaluation trials verification
    eval_cfg = cfg.get("evaluation", {})
    eval_dev = bool(eval_cfg.get("eval_dev", True))
    eval_probe = bool(eval_cfg.get("eval_train_probe", True))

    if eval_dev:
        dev_trials = get_data_root() / "trials" / "trials_dev-clean.txt"
        if not dev_trials.exists():
            raise FileNotFoundError(f"Requested eval_dev but trial file not found: {dev_trials}")
    else:
        dev_trials = None

    if eval_probe:
        train_probe_trials = get_data_root() / "trials" / "trials_train-probe.txt"
        if not train_probe_trials.exists():
            raise FileNotFoundError(f"Requested eval_train_probe but trial file not found: {train_probe_trials}")
    else:
        train_probe_trials = None

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        scheduler=scheduler,
        device=device,
        grad_clip=float(cfg.training.get("grad_clip", 3.0)),
        run_id=run_id,
        config_hash=config_hash,
        dev_trials_path=dev_trials,
        train_probe_trials_path=train_probe_trials,
        precision=args.precision,
        seed=seed,
    )

    start_epoch = 1
    if args.resume:
        latest_ckpt = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt is not None:
            print(f"Resuming training from checkpoint: {latest_ckpt}")
            start_epoch = trainer.load_checkpoint(latest_ckpt)
        else:
            print(f"No previous checkpoint found in {ckpt_dir}. Starting from scratch.")

    if cfg.training.get("overfit_batch", False):
        steps = int(cfg.training.get("overfit_steps", 300))
        trainer.fit_overfit_batch(num_steps=steps)
        return 0

    eval_interval = int(cfg.training.get("eval_interval", 1))
    trainer.fit(num_epochs=num_epochs, start_epoch=start_epoch, eval_interval=eval_interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())

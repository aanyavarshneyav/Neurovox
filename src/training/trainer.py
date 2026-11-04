"""Modular training loop for speaker verification with NaN guards and stateful resume parity.

Features:
- Adam optimizer with triangular CyclicLR schedule (cycle_momentum=False).
- NaN/Inf detection at every step with descriptive FloatingPointError crashes.
- Full-utterance validation on dev-clean (and optional train-probe) with separate timing.
- Epoch-by-epoch history export (JSON & CSV artifacts).
- Full stateful checkpointing (model, loss, optimizer, scheduler, 4 RNGs) for bitwise resume parity.
- Retention management keeping the last 2 epoch_*.pt, best_dev_eer.pt, and final.pt.
"""

import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.extractor import EmbeddingExtractor
from src.eval.metrics import compute_eer, compute_mindcf
from src.utils.paths import get_ckpt_root, get_results_root, resolve_path

logger = logging.getLogger(__name__)


def get_git_info() -> Tuple[str, bool]:
    """Retrieve Git commit SHA and dirty working tree status."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return sha, bool(status)
    except Exception:
        return "unknown", False


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Find the latest epoch_*.pt checkpoint in ckpt_dir sorted by epoch number."""
    ckpt_dir = resolve_path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    ckpts = [p for p in ckpt_dir.glob("epoch_*.pt") if p.stem.split("_")[1].isdigit()]
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.stem.split("_")[1]))



class Trainer:
    """Speaker verification trainer supporting stateful resume, NaN guards, and full-utterance evaluation."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        grad_clip: float = 3.0,
        run_id: str = "run_default",
        config_hash: str = "unknown",
        dev_trials_path: Optional[Union[str, Path]] = None,
        train_probe_trials_path: Optional[Union[str, Path]] = None,
        precision: str = "fp32",
        seed: int = 42,
        keep_last_n: int = 2,
    ) -> None:
        """Initialize Trainer.

        Args:
            model: Speaker embedding model.
            loss_fn: Margin loss module (e.g. AAMSoftmax).
            optimizer: PyTorch optimizer (e.g. Adam).
            train_loader: DataLoader yielding training batches.
            scheduler: Learning rate scheduler (e.g. CyclicLR).
            device: Compute device.
            grad_clip: Maximum gradient norm for clipping.
            run_id: Unique experiment run identifier.
            config_hash: Hash of configuration YAML.
            dev_trials_path: Path to evaluation trial file for dev-clean.
            train_probe_trials_path: Path to train-probe trial file (for training-data EER).
            precision: Precision mode ('fp32').
            seed: Base random seed.
            keep_last_n: Number of most recent epoch_*.pt checkpoints to retain.
        """
        if precision != "fp32":
            raise ValueError(f"Trainer only supports precision='fp32', got '{precision}'")

        self.device = device if device is not None else torch.device("cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.grad_clip = grad_clip
        self.run_id = run_id
        self.config_hash = config_hash
        self.dev_trials_path = resolve_path(dev_trials_path) if dev_trials_path else None
        self.train_probe_trials_path = (
            resolve_path(train_probe_trials_path) if train_probe_trials_path else None
        )
        self.precision = precision
        self.seed = seed
        self.keep_last_n = keep_last_n

        self.history: List[Dict[str, Any]] = []
        self.seen_utterances: Set[int] = set()
        self.best_dev_eer = float("inf")
        self.step_0_loss: Optional[float] = None
        self.epoch_0_dev_eer: Optional[float] = None
        self.epoch_0_train_probe_eer: Optional[float] = None

        # Directories
        self.ckpt_dir = get_ckpt_root() / self.run_id
        self.results_dir = get_results_root() / self.run_id
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._write_run_meta()

    def _write_run_meta(self) -> None:
        """Write results/<run_id>/run_meta.json at initialization."""
        git_sha, git_dirty = get_git_info()
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        ds = getattr(self.train_loader, "dataset", None)
        aug_flags = {
            "speed": getattr(ds, "speed_aug", None) is not None,
            "noise": getattr(ds, "noise_aug", None) is not None,
            "babble": getattr(ds, "babble_aug", None) is not None,
            "rir": getattr(ds, "rir_aug", None) is not None,
            "specaug": getattr(ds, "enable_specaug", False),
        }
        aug_probs = getattr(ds, "augmentation_probabilities", {})

        batch_sampler = getattr(self.train_loader, "batch_sampler", None)
        batch_size = getattr(batch_sampler, "batch_size", getattr(self.train_loader, "batch_size", 0))
        num_speakers = getattr(batch_sampler, "num_speakers", 0)

        peak_gpu_mem = (
            int(torch.cuda.max_memory_allocated(self.device))
            if torch.cuda.is_available() and self.device.type == "cuda"
            else (int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0)
        )

        lr_info: Dict[str, Any] = {
            "type": type(self.scheduler).__name__ if self.scheduler is not None else "constant",
        }
        if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.CyclicLR):
            lr_info.update({
                "base_lrs": self.scheduler.base_lrs,
                "max_lrs": self.scheduler.max_lrs,
                "total_size": getattr(self.scheduler, "total_size", None),
                "step_size_up": getattr(self.scheduler, "step_size_up", int(self.scheduler.total_size * getattr(self.scheduler, "step_ratio", 0.5))),
                "mode": self.scheduler.mode,
            })
        else:
            lr_info["lr"] = self.optimizer.param_groups[0]["lr"]

        meta = {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "device": str(self.device),
            "precision": self.precision,
            "augmentation_flags": aug_flags,
            "augmentation_probabilities": aug_probs,
            "torch_version": torch.__version__,
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "parameter_count": param_count,
            "trainable_parameter_count": trainable_param_count,
            "num_workers": getattr(self.train_loader, "num_workers", 0),
            "peak_gpu_memory_bytes": peak_gpu_mem,
            "batch_size": batch_size,
            "batches_per_epoch": len(self.train_loader) if hasattr(self.train_loader, "__len__") else 0,
            "grad_clip": self.grad_clip,
            "lr_schedule": lr_info,
            "dataset_size": len(ds) if ds is not None else 0,
            "num_speakers": num_speakers,
        }

        meta_path = self.results_dir / "run_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _manage_checkpoints(self) -> None:
        """Retain only the last `keep_last_n` epoch_*.pt checkpoints."""
        epoch_ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        if len(epoch_ckpts) > self.keep_last_n:
            for ckpt in epoch_ckpts[: -self.keep_last_n]:
                try:
                    ckpt.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove older checkpoint {ckpt}: {e}")

    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        """Execute one complete training epoch over train_loader."""
        self.model.train()
        self.loss_fn.train()
        if hasattr(self.loss_fn, "set_epoch"):
            self.loss_fn.set_epoch(epoch)
        if hasattr(self.loss_fn, "reset_epoch_diagnostics"):
            self.loss_fn.reset_epoch_diagnostics()

        # Update batch_sampler epoch
        batch_sampler = getattr(self.train_loader, "batch_sampler", None)
        if batch_sampler is None or not hasattr(batch_sampler, "set_epoch"):
            raise TypeError("train_loader.batch_sampler must have a set_epoch method")
        batch_sampler.set_epoch(epoch)

        total_loss_tensor = torch.zeros(1, device=self.device, dtype=torch.float32)
        total_correct_tensor = torch.zeros(1, device=self.device, dtype=torch.long)
        total_samples_tensor = torch.zeros(1, device=self.device, dtype=torch.long)
        grad_norms_sum_tensor = torch.zeros(1, device=self.device, dtype=torch.float32)

        data_loading_sec = 0.0
        forward_backward_optimizer_sec = 0.0

        t0_train = time.perf_counter()
        t_prev = time.perf_counter()

        for batch_idx, batch_data in enumerate(self.train_loader, start=1):
            t_data_ready = time.perf_counter()
            data_loading_sec += (t_data_ready - t_prev)

            t0_step = time.perf_counter()
            if isinstance(batch_data, dict):
                data = batch_data.get("features", batch_data.get("waveform"))
                labels = batch_data["label"]
            else:
                data, labels = batch_data

            data = data.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            # Check label sanity on first batch of epoch 1
            if epoch == 1 and batch_idx == 1:
                num_classes = getattr(self.loss_fn, "num_classes", None)
                if num_classes is not None:
                    min_lbl = labels.min().item()
                    max_lbl = labels.max().item()
                    if min_lbl < 0 or max_lbl >= num_classes:
                        raise ValueError(
                            f"Target label out of bounds: [{min_lbl}, {max_lbl}], "
                            f"expected [0, {num_classes - 1}]"
                        )

            # Stage 1: Input features NaN/Inf check
            if not torch.isfinite(data).all():
                raise FloatingPointError(
                    f"[NaN DETECTED] Stage: input_features | Batch: {batch_idx} | Run: {self.run_id}"
                )

            self.optimizer.zero_grad(set_to_none=True)

            # Stage 2: Forward pass & embedding NaN/Inf check
            emb = self.model(data)
            if not torch.isfinite(emb).all():
                raise FloatingPointError(
                    f"[NaN DETECTED] Stage: model_embeddings | Batch: {batch_idx} | Run: {self.run_id}"
                )

            # Stage 3: Compute margin loss & NaN/Inf check
            loss = self.loss_fn(emb, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"[NaN DETECTED] Stage: loss_computation | Batch: {batch_idx} | Run: {self.run_id}"
                )

            # On-device loss & accuracy accumulation
            total_loss_tensor += loss.detach()
            with torch.no_grad():
                logits = (
                    self.loss_fn.get_logits(emb)
                    if hasattr(self.loss_fn, "get_logits")
                    else emb
                )
                preds = torch.argmax(logits, dim=1)
                total_correct_tensor += (preds == labels).sum()
                total_samples_tensor += labels.size(0)

            # Backward pass
            loss.backward()

            # Stage 4: Gradient check and clipping via clip_grad_norm_
            max_norm = self.grad_clip if self.grad_clip > 0 else float("inf")
            all_params = list(self.model.parameters()) + list(self.loss_fn.parameters())
            total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_norm)

            if not torch.isfinite(total_norm):
                offender = None
                for name, param in list(self.model.named_parameters()) + list(self.loss_fn.named_parameters()):
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        offender = name
                        break
                raise FloatingPointError(
                    f"[NaN DETECTED] Stage: grad_{offender or 'unknown'} | Batch: {batch_idx} | Run: {self.run_id}"
                )

            grad_norms_sum_tensor += total_norm

            self.optimizer.step()

            # Step cyclic scheduler per batch
            if self.scheduler is not None and isinstance(
                self.scheduler, torch.optim.lr_scheduler.CyclicLR
            ):
                self.scheduler.step()

            forward_backward_optimizer_sec += (time.perf_counter() - t0_step)
            t_prev = time.perf_counter()

        train_sec = time.perf_counter() - t0_train

        # Synchronize accumulated loss, accuracy, and gradient norm once per epoch
        avg_loss = float(total_loss_tensor.item()) / max(1, len(self.train_loader))
        num_samples = int(total_samples_tensor.item())
        train_acc = (float(total_correct_tensor.item()) / max(1, num_samples)) * 100.0
        avg_grad_norm = float(grad_norms_sum_tensor.item()) / max(1, len(self.train_loader))
        current_lr = self.optimizer.param_groups[0]["lr"]

        # Track sampler indices for the epoch
        epoch_indices = getattr(batch_sampler, "current_epoch_indices", None)
        if epoch_indices and len(epoch_indices) > 0:
            first_batch = list(epoch_indices[0])
            first_batch_hash = hashlib.sha256(str(first_batch).encode("utf-8")).hexdigest()[:16]
            epoch_utts = set(int(idx) for batch in epoch_indices for idx in batch)
            unique_utts_this_epoch = len(epoch_utts)
            self.seen_utterances.update(epoch_utts)
            cum_unique_utts = len(self.seen_utterances)
        else:
            first_batch_hash = "none"
            unique_utts_this_epoch = 0
            cum_unique_utts = len(self.seen_utterances)

        epoch_metrics = {
            "train_loss": avg_loss,
            "train_loss_unrounded": avg_loss,
            "train_acc": round(train_acc, 2),
            "train_sec": round(train_sec, 2),
            "data_loading_sec": round(data_loading_sec, 2),
            "forward_backward_optimizer_sec": round(forward_backward_optimizer_sec, 2),
            "grad_norm": round(avg_grad_norm, 4),
            "lr": current_lr,
            "sampler_epoch": epoch,
            "first_batch_index_hash": first_batch_hash,
            "unique_utts_this_epoch": unique_utts_this_epoch,
            "cum_unique_utts": cum_unique_utts,
        }

        if hasattr(self.loss_fn, "get_epoch_diagnostics"):
            epoch_metrics.update(self.loss_fn.get_epoch_diagnostics())

        return epoch_metrics

    def evaluate_trials(self, trial_file: Path) -> Tuple[float, float, float]:
        """Extract full-utterance embeddings and compute EER and minDCF (0.05)."""
        extractor = EmbeddingExtractor(self.model, device=self.device, batch_size=1)
        t0_eval = time.perf_counter()

        embs = extractor.extract_for_trial_file(trial_file)

        labels: List[int] = []
        scores: List[float] = []

        with open(trial_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    lbl = int(parts[0])
                    ea = embs[parts[1]]
                    eb = embs[parts[2]]
                    score = float(np.dot(ea, eb))
                    labels.append(lbl)
                    scores.append(score)

        eval_sec = time.perf_counter() - t0_eval
        scores_arr = np.array(scores, dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.int32)

        eer, _ = compute_eer(scores_arr, labels_arr)
        mindcf_05 = compute_mindcf(scores_arr, labels_arr, p_target=0.05, normalize=True)

        return float(eer) * 100.0, float(mindcf_05.min_dcf), eval_sec

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Save full training state including optimizer, scheduler, history, and RNGs."""
        rng_state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            rng_state["mps"] = torch.mps.get_rng_state()

        payload = {
            "epoch": epoch,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "precision": self.precision,
            "device": str(self.device),
            "model_state_dict": self.model.state_dict(),
            "loss_state_dict": self.loss_fn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "rng_state": rng_state,
            "history": self.history,
            "seen_utterances": list(self.seen_utterances),
            "best_dev_eer": self.best_dev_eer,
            "step_0_loss": self.step_0_loss,
            "epoch_0_dev_eer": self.epoch_0_dev_eer,
            "epoch_0_train_probe_eer": self.epoch_0_train_probe_eer,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: Path) -> int:
        """Restore full training state and return the next epoch index to run."""
        path = resolve_path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)

        # Enforce strict configuration, precision, and device consistency
        ckpt_hash = payload.get("config_hash", "")
        if ckpt_hash and self.config_hash and self.config_hash != "unknown" and ckpt_hash != self.config_hash:
            raise ValueError(
                f"Config hash mismatch on resume: ckpt={ckpt_hash} vs current={self.config_hash}"
            )

        ckpt_precision = payload.get("precision", "fp32")
        if ckpt_precision != self.precision:
            raise ValueError(
                f"Precision mismatch on resume: ckpt={ckpt_precision} vs current={self.precision}"
            )

        ckpt_device = payload.get("device", str(self.device))
        if ckpt_device != str(self.device):
            raise ValueError(
                f"Device mismatch on resume: ckpt={ckpt_device} vs current={self.device}"
            )

        self.model.load_state_dict(payload["model_state_dict"])
        self.loss_fn.load_state_dict(payload["loss_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])

        if self.scheduler is not None and payload.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])

        # Restore RNG states
        rng = payload.get("rng_state", {})
        if "torch" in rng:
            torch_state = rng["torch"]
            if isinstance(torch_state, torch.Tensor):
                torch_state = torch_state.cpu()
            torch.set_rng_state(torch_state)
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "python" in rng:
            random.setstate(rng["python"])
        if ("cuda" in rng or "cuda_rng_state" in payload) and torch.cuda.is_available():
            cuda_state = rng.get("cuda", payload.get("cuda_rng_state"))
            if isinstance(cuda_state, torch.Tensor):
                cuda_state = cuda_state.cpu()
            torch.cuda.set_rng_state(cuda_state)
        elif "mps" in rng and hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                mps_state = rng["mps"]
                if isinstance(mps_state, torch.Tensor):
                    mps_state = mps_state.cpu()
                torch.mps.set_rng_state(mps_state)
            except Exception as e:
                logger.warning(f"Could not restore MPS RNG state: {e}")

        # Restore history and seen utterances
        self.history = payload.get("history", [])
        if not self.history:
            hist_file = self.results_dir / "history.json"
            if hist_file.exists():
                with open(hist_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)

        self.seen_utterances = set(payload.get("seen_utterances", []))
        self.best_dev_eer = payload.get("best_dev_eer", float("inf"))
        self.step_0_loss = payload.get("step_0_loss", None)
        self.epoch_0_dev_eer = payload.get("epoch_0_dev_eer", None)
        self.epoch_0_train_probe_eer = payload.get("epoch_0_train_probe_eer", None)

        start_epoch = payload.get("epoch", 0) + 1
        if hasattr(self.loss_fn, "set_epoch"):
            self.loss_fn.set_epoch(start_epoch)
        return start_epoch

    def save_history_artifacts(self) -> None:
        """Save training history as JSON and CSV artifacts."""
        json_path = self.results_dir / "history.json"
        csv_path = self.results_dir / "history.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        if self.history:
            # Collect unique non-nested keys across all records for clean CSV export
            keys = []
            seen_keys = set()
            for r in self.history:
                for k, v in r.items():
                    if k not in seen_keys and not isinstance(v, (dict, list)):
                        seen_keys.add(k)
                        keys.append(k)
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.history)

    def fit(
        self,
        num_epochs: int,
        start_epoch: int = 1,
        eval_interval: int = 1,
    ) -> List[Dict[str, Any]]:
        """Execute full training loop from start_epoch to num_epochs."""
        print("=" * 95)
        print(f"TRAINING START: Run ID: {self.run_id} | Device: {self.device} | Precision: {self.precision}")
        print(f"Epochs: {start_epoch} -> {num_epochs} | Batches/Epoch: {len(self.train_loader)}")
        print("=" * 95)

        # Step 0 initial loss and epoch 0 baseline evaluation
        if start_epoch == 1:
            self.model.eval()
            self.loss_fn.eval()
            with torch.no_grad():
                first_batch = next(iter(self.train_loader))
                if isinstance(first_batch, dict):
                    f_data = first_batch.get("features", first_batch.get("waveform"))
                    f_labels = first_batch["label"]
                else:
                    f_data, f_labels = first_batch
                f_data = f_data.to(self.device, dtype=torch.float32)
                f_labels = f_labels.to(self.device, dtype=torch.long)
                emb0 = self.model(f_data)
                self.step_0_loss = round(float(self.loss_fn(emb0, f_labels).item()), 4)

            if self.dev_trials_path:
                dev_eer0, dev_mindcf0, sec0 = self.evaluate_trials(self.dev_trials_path)
                self.epoch_0_dev_eer = round(dev_eer0, 2)
                probe_str0 = ""
                if self.train_probe_trials_path and self.train_probe_trials_path.exists():
                    probe_eer0, _, _ = self.evaluate_trials(self.train_probe_trials_path)
                    self.epoch_0_train_probe_eer = round(probe_eer0, 2)
                    probe_str0 = f" | train_probe_eer: {self.epoch_0_train_probe_eer:.2f}%"

                print(
                    f"[Epoch 00 (untrained)] step_0_loss: {self.step_0_loss:.4f} | "
                    f"dev_eer: {self.epoch_0_dev_eer:.2f}% (minDCF: {dev_mindcf0:.4f}){probe_str0}"
                )

        for epoch in range(start_epoch, num_epochs + 1):
            t0_epoch = time.perf_counter()
            train_metrics = self.train_epoch(epoch)

            peak_gpu_mem = (
                int(torch.cuda.max_memory_allocated(self.device))
                if torch.cuda.is_available() and self.device.type == "cuda"
                else (int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0)
            )
            num_workers = getattr(self.train_loader, "num_workers", 0)

            record: Dict[str, Any] = {
                "epoch": epoch,
                "sampler_epoch": train_metrics["sampler_epoch"],
                "first_batch_index_hash": train_metrics["first_batch_index_hash"],
                "unique_utts_this_epoch": train_metrics["unique_utts_this_epoch"],
                "cum_unique_utts": train_metrics["cum_unique_utts"],
                "train_loss": round(train_metrics["train_loss"], 4),
                "train_loss_unrounded": train_metrics["train_loss"],
                "train_acc": train_metrics["train_acc"],
                "train_sec": train_metrics["train_sec"],
                "data_loading_sec": train_metrics["data_loading_sec"],
                "forward_backward_optimizer_sec": train_metrics["forward_backward_optimizer_sec"],
                "lr": train_metrics["lr"],
                "grad_norm": train_metrics["grad_norm"],
                "eval_sec": 0.0,
                "dev_clean_eval_sec": 0.0,
                "dev_eer_pct": None,
                "dev_mindcf_05": None,
                "train_probe_eer_pct": None,
                "peak_gpu_memory_bytes": peak_gpu_mem,
                "num_workers": num_workers,
            }

            # Forward loss diagnostics (e.g. AdaptiveDrop) into epoch history record
            for k, v in train_metrics.items():
                if k.startswith("loss_"):
                    record[k] = v

            # Periodic full-utterance evaluation
            if (epoch % eval_interval == 0 or epoch == num_epochs) and self.dev_trials_path:
                dev_eer, dev_mindcf, dev_eval_sec = self.evaluate_trials(self.dev_trials_path)
                record["dev_eer_pct"] = round(dev_eer, 2)
                record["dev_mindcf_05"] = round(dev_mindcf, 4)
                record["dev_clean_eval_sec"] = round(dev_eval_sec, 2)
                record["eval_sec"] = round(dev_eval_sec, 2)

                # Train-probe evaluation (training-data EER)
                if self.train_probe_trials_path and self.train_probe_trials_path.exists():
                    probe_eer, _, probe_sec = self.evaluate_trials(self.train_probe_trials_path)
                    record["train_probe_eer_pct"] = round(probe_eer, 2)
                    record["eval_sec"] = round(record["eval_sec"] + probe_sec, 2)

            total_sec = time.perf_counter() - t0_epoch
            record["total_sec"] = round(total_sec, 2)

            # Append current record BEFORE saving best_dev_eer.pt (F10)
            self.history.append(record)

            if record["dev_eer_pct"] is not None and record["dev_eer_pct"] < self.best_dev_eer:
                self.best_dev_eer = record["dev_eer_pct"]
                self.save_checkpoint(self.ckpt_dir / "best_dev_eer.pt", epoch)

            # Epoch checkpoint with retention
            self.save_checkpoint(self.ckpt_dir / f"epoch_{epoch:03d}.pt", epoch)
            self._manage_checkpoints()
            self.save_history_artifacts()

            # Output epoch summary
            dev_str = (
                f"dev_eer: {record['dev_eer_pct']:.2f}% | minDCF: {record['dev_mindcf_05']:.4f}"
                if record["dev_eer_pct"] is not None
                else "eval: skipped"
            )
            probe_str = (
                f"train_probe_eer (training-data): {record['train_probe_eer_pct']:.2f}%"
                if record["train_probe_eer_pct"] is not None
                else ""
            )

            ad_str = ""
            if "loss_keep_ratio" in record:
                ad_str = (
                    f" | keep: {record['loss_keep_ratio']:.4f} "
                    f"(ret: {record.get('loss_num_retained', 0)}, "
                    f"filt: {record.get('loss_num_filtered', 0)}, "
                    f"corr: {record.get('loss_num_corrected', 0)}, "
                    f"guard: {record.get('loss_guard_activations', 0)})"
                )

            print(
                f"[Epoch {epoch:02d}/{num_epochs:02d}] "
                f"loss: {record['train_loss']:.4f} | "
                f"acc: {record['train_acc']:.1f}% | "
                f"lr: {record['lr']:.2e} | "
                f"train_time: {record['train_sec']:.1f}s | "
                f"eval_time: {record['eval_sec']:.1f}s | "
                f"{dev_str} | {probe_str}{ad_str}"
            )

        # Final checkpoint
        self.save_checkpoint(self.ckpt_dir / "final.pt", num_epochs)
        self._write_run_meta()

        print("=" * 95)
        print(f"TRAINING COMPLETE: {self.run_id} | Best Dev EER: {self.best_dev_eer:.2f}%")
        print("=" * 95)
        return self.history

    def fit_overfit_batch(
        self,
        num_steps: int = 300,
        epoch: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Overfit a single fixed batch to verify numerical stability and optimization dynamics.

        Args:
            num_steps: Number of optimization steps on the fixed batch.
            epoch: Optional explicit 1-indexed epoch for loss functions with epoch-dependent
                   schedules (e.g. IAAMSoftmax with AdaptiveDrop). If None, the existing
                   loss epoch state is preserved (defaulting to 1 / warm-up).

        Gate: 100% train accuracy and loss reduced >= 100x, zero NaNs.
        """
        self.model.train()
        self.loss_fn.train()
        if epoch is not None and hasattr(self.loss_fn, "set_epoch"):
            self.loss_fn.set_epoch(epoch)

        # Fetch first batch
        batch = next(iter(self.train_loader))
        if isinstance(batch, dict):
            data = batch.get("features", batch.get("waveform"))
            labels = batch["label"]
        else:
            data, labels = batch

        data = data.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device, dtype=torch.long)

        current_ep = getattr(self.loss_fn, "current_epoch", 1)
        print("=" * 80)
        print(f"SANITY RUN: Overfitting Single Fixed Batch ({data.size(0)} samples)")
        print(f"Device: {self.device} | Precision: {self.precision} | Steps: {num_steps} | Epoch: {current_ep}")
        print("=" * 80)

        initial_loss = None
        history = []

        t0 = time.perf_counter()
        for step in range(1, num_steps + 1):
            if not torch.isfinite(data).all():
                raise FloatingPointError(
                    f"[NaN DETECTED] Stage: input_features | Step: {step} | Run: {self.run_id}"
                )

            self.optimizer.zero_grad(set_to_none=True)

            emb = self.model(data)
            if not torch.isfinite(emb).all():
                raise FloatingPointError(
                    f"[NaN DETECTED] Stage: model_embeddings | Step: {step} | Run: {self.run_id}"
                )

            loss = self.loss_fn(emb, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"[NaN DETECTED] Stage: loss_computation | Step: {step} | Run: {self.run_id}"
                )

            loss_val = float(loss.item())
            if initial_loss is None:
                initial_loss = loss_val

            # Cosine accuracy without margin
            with torch.no_grad():
                logits = (
                    self.loss_fn.get_logits(emb)
                    if hasattr(self.loss_fn, "get_logits")
                    else emb
                )
                preds = torch.argmax(logits, dim=1)
                acc = float((preds == labels).sum().item()) / labels.size(0) * 100.0
                keep_mask = getattr(self.loss_fn, "last_keep_mask", None)
                if keep_mask is not None and keep_mask.shape == labels.shape:
                    retained_cnt = max(1, int(keep_mask.sum().item()))
                    eval_acc = float((preds[keep_mask] == labels[keep_mask]).sum().item()) / retained_cnt * 100.0
                else:
                    eval_acc = acc

            loss.backward()

            max_norm = self.grad_clip if self.grad_clip > 0 else float("inf")
            all_params = list(self.model.parameters()) + list(self.loss_fn.parameters())
            norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_norm)

            if not torch.isfinite(norm):
                offender = None
                for name, param in list(self.model.named_parameters()) + list(self.loss_fn.named_parameters()):
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        offender = name
                        break
                raise FloatingPointError(
                    f"[NaN DETECTED] Stage: grad_{offender or 'unknown'} | Step: {step} | Run: {self.run_id}"
                )

            self.optimizer.step()

            record = {
                "step": step,
                "loss": round(loss_val, 6),
                "acc_pct": round(acc, 2),
                "retained_acc_pct": round(eval_acc, 2),
                "grad_norm": round(float(norm.item() if isinstance(norm, torch.Tensor) else norm), 4),
            }
            if hasattr(self.loss_fn, "get_last_diagnostics"):
                diag = self.loss_fn.get_last_diagnostics()
                if diag:
                    record["loss_diagnostics"] = diag
            history.append(record)

            if step == 1 or step % 25 == 0 or step == num_steps:
                elapsed = time.perf_counter() - t0
                diag_str = ""
                if hasattr(self.loss_fn, "get_last_diagnostics"):
                    d = self.loss_fn.get_last_diagnostics()
                    if d:
                        diag_str = (
                            f" | keep: {d.get('num_retained', 0)}/{d.get('batch_size', 0)} "
                            f"(corr: {d.get('num_corrected', 0)}, drop: {d.get('num_filtered', 0)}, guard: {d.get('guard_activated', False)})"
                        )
                acc_label = f"Acc: {eval_acc:6.2f}%" if eval_acc == acc else f"Acc(ret): {eval_acc:5.1f}% [all:{acc:4.1f}%]"
                print(
                    f"Step {step:03d}/{num_steps} | Loss: {loss_val:.4f} | "
                    f"{acc_label}{diag_str} | GradNorm: {record['grad_norm']:.3f} | Elapsed: {elapsed:.1f}s"
                )

        self.history = history
        self.save_history_artifacts()
        final_ep = epoch if epoch is not None else getattr(self.loss_fn, "current_epoch", 1)
        self.save_checkpoint(self.ckpt_dir / "overfit_final.pt", final_ep)

        # Technical gate check
        loss_reduction = initial_loss / max(loss_val, 1e-12)
        print("-" * 80)
        print(f"Initial Loss: {initial_loss:.4f} -> Final Loss: {loss_val:.6f} (reduction: {loss_reduction:.1f}x)")
        print(f"Final Accuracy: {acc:.2f}% (retained: {eval_acc:.2f}%)")
        if hasattr(self.loss_fn, "get_last_diagnostics"):
            final_diag = self.loss_fn.get_last_diagnostics()
            if final_diag:
                print(
                    f"Final Loss Diagnostics (Step {num_steps}, Epoch {final_ep}): "
                    f"Retained={final_diag.get('num_retained')}/{final_diag.get('batch_size')}, "
                    f"Corrected={final_diag.get('num_corrected')}, "
                    f"Filtered={final_diag.get('num_filtered')}, "
                    f"KeepRatio={final_diag.get('keep_ratio')}, "
                    f"GuardActive={final_diag.get('guard_activated')}"
                )
        print("-" * 80)

        if eval_acc < 100.0:
            raise RuntimeError(f"Gate Failed: Final retained accuracy {eval_acc:.2f}% < 100.0%")
        if loss_reduction < 100.0:
            raise RuntimeError(f"Gate Failed: Loss reduction {loss_reduction:.1f}x < 100.0x")

        print(">>> TECHNICAL GATE PASSED: 100% train accuracy, loss reduced >= 100x, zero NaNs.")
        return history


__all__ = ["Trainer"]

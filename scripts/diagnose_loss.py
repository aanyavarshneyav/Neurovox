#!/usr/bin/env python3
"""Diagnostic script for IAAMSoftmax and AdaptiveDrop inspection (Phase 2C).

Demonstrates and exposes:
- Number of corrected labels.
- Number of filtered samples.
- Actual keep ratio.
- Whether the minimum keep-ratio guard activated.
- Loss value.

Can be run standalone:
    python scripts/diagnose_loss.py
"""

import argparse
import math
import sys
from pathlib import Path

import torch

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.losses.iaam_softmax import IAAMSoftmax


def run_diagnostic(
    batch_size: int = 64,
    num_classes: int = 251,
    emb_dim: int = 192,
    noise_rate: float = 0.25,
    epoch: int = 6,
    min_keep_ratio: float = 0.5,
    device_str: str = "auto",
) -> None:
    if device_str == "auto":
        device = (
            torch.device("mps")
            if hasattr(torch, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(device_str)

    print("=" * 70)
    print("IAAMSoftmax & AdaptiveDrop Diagnostic Run")
    print("=" * 70)
    print(f"Device:           {device}")
    print(f"Batch size:       {batch_size}")
    print(f"Classes:          {num_classes} | Embedding Dim: {emb_dim}")
    print(f"Current Epoch:    {epoch} (ESD=5, ESC=5)")
    print(f"Label Noise:      {noise_rate * 100:.1f}%")
    print(f"Min Keep Ratio:   {min_keep_ratio}")
    print("-" * 70)

    torch.manual_seed(42)
    loss_fn = IAAMSoftmax(
        in_features=emb_dim,
        num_classes=num_classes,
        scale=30.0,
        margin=0.2,
        tau=0.423,
        tau_corr=0.5,
        esd=5,
        esc=5,
        min_keep_ratio=min_keep_ratio,
    ).to(device)

    loss_fn.train()
    loss_fn.set_epoch(epoch)

    # Synthesize embeddings aligned with prototype vectors plus Gaussian noise
    true_labels = torch.randint(0, num_classes, (batch_size,), device=device)
    embeddings = []
    with torch.no_grad():
        prototypes = loss_fn.weight.clone()
        for lbl in true_labels:
            # 80% alignment with true class prototype + random perturbation
            proto = prototypes[lbl]
            noise = torch.randn_like(proto) * 0.3
            v = proto + noise
            embeddings.append(v)
    x = torch.stack(embeddings).to(device)
    x.requires_grad_(True)

    # Corrupt a fraction of labels to simulate annotation error or clustering noise
    noisy_labels = true_labels.clone()
    num_corrupt = int(round(batch_size * noise_rate))
    corrupt_idx = torch.randperm(batch_size)[:num_corrupt]
    for idx in corrupt_idx:
        # Reassign to random competing class
        competing = (noisy_labels[idx] + torch.randint(1, num_classes, (1,), device=device)) % num_classes
        noisy_labels[idx] = competing

    # Execute forward pass with diagnostic return
    loss, diag = loss_fn(x, noisy_labels, return_diagnostics=True)

    # Backward pass to ensure gradient propagation
    loss.backward()

    print("\nDiagnostic Results:")
    print(f"  • Loss Value:                      {diag['loss']:.6f}")
    print(f"  • Retained Samples:                {diag['num_retained']} / {diag['batch_size']}")
    print(f"  • Filtered (Dropped) Samples:      {diag['num_filtered']}")
    print(f"  • Corrected Labels:                {diag['num_corrected']}")
    print(f"  • Actual Keep Ratio:               {diag['keep_ratio']:.4f}")
    print(f"  • Minimum Keep-Ratio Guard Active: {diag['guard_activated']}")
    print(f"  • Mean Target Cosine Similarity:   {diag['mean_target_cs']:.4f}")
    print(f"  • Embedding Gradient Norm:         {x.grad.norm().item():.4f}")
    print(f"  • Prototype Gradient Norm:         {loss_fn.weight.grad.norm().item():.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IAAMSoftmax Diagnostic Script")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=6)
    parser.add_argument("--noise-rate", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--min-keep-ratio", type=float, default=0.5)
    args = parser.parse_args()

    run_diagnostic(
        batch_size=args.batch_size,
        epoch=args.epoch,
        noise_rate=args.noise_rate,
        device_str=args.device,
        min_keep_ratio=args.min_keep_ratio,
    )

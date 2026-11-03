#!/usr/bin/env python3
"""Parameter audit script for Neurovox architectures (Phase 2C).

Inspects layer-by-layer and total parameters using torchinfo.summary():
- Scaled SKA-HNN S0 / S1 (C=32, C1=256)
- Ablation variants S2, S3, S4, S5, S6
- Parameter-matched ECAPA-TDNN baseline (C=400, MFA=1200)
- Standard ECAPA-TDNN baseline (C=1024, MFA=3072)
- Small TDNN-SE baseline (Phase 2A sanity architecture)

Conforms to .agents/workflows/param-audit.md.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch
from torchinfo import summary

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models import (
    ECAPATDNN,
    SKAHNN,
    SmallTDNNSE,
    build_ska_hnn,
    ecapa_tdnn_param_matched,
    ecapa_tdnn_standard,
    small_tdnn_se,
)


def run_audit(device: str = "cpu", input_shape: tuple = (1, 80, 180)) -> Dict[str, Dict[str, Any]]:
    dev = torch.device(device)
    print("=" * 90)
    print("NEUROVOX ARCHITECTURE PARAMETER AUDIT (TORCHINFO)")
    print(f"Device: {dev} | Input Shape: {input_shape}")
    print("=" * 90)

    results: Dict[str, Dict[str, Any]] = {}

    models_to_audit = [
        ("Scaled SKA-HNN (S0)", build_ska_hnn(switch="S0")),
        ("Scaled SKA-HNN (S1)", build_ska_hnn(switch="S1")),
        ("Ablation S2 (No MLASP Context)", build_ska_hnn(switch="S2")),
        ("Ablation S3 (No MLFC)", build_ska_hnn(switch="S3")),
        ("Ablation S4 (No MLFC & No Context)", build_ska_hnn(switch="S4")),
        ("Ablation S5 (Vanilla 2D-CNN)", build_ska_hnn(switch="S5")),
        ("Ablation S6 (No 2D-CNN Frontend)", build_ska_hnn(switch="S6")),
        ("Parameter-Matched ECAPA-TDNN", ecapa_tdnn_param_matched()),
        ("Standard ECAPA-TDNN (C=1024)", ecapa_tdnn_standard()),
        ("Small TDNN-SE (Phase 2A)", small_tdnn_se()),
    ]

    print(f"{'Model / Switch ID':<35} | {'Total Params':<14} | {'Trainable':<14} | {'Mult-Adds':<12}")
    print("-" * 90)

    for name, model in models_to_audit:
        model = model.to(dev)
        s = summary(model, input_size=input_shape, verbose=0, device=dev)
        mult_adds_str = f"{s.total_mult_adds / 1e6:.2f}M" if s.total_mult_adds else "N/A"
        print(f"{name:<35} | {s.total_params:<14,d} | {s.trainable_params:<14,d} | {mult_adds_str:<12}")
        results[name] = {
            "total_params": s.total_params,
            "trainable_params": s.trainable_params,
            "non_trainable_params": s.total_params - s.trainable_params,
            "mult_adds": s.total_mult_adds,
        }

    print("=" * 90)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Neurovox model parameters")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--input_shape", nargs="+", type=int, default=[1, 80, 180])
    args = parser.parse_args()

    shape = tuple(args.input_shape)
    if len(shape) == 4 and shape[1] == 1:
        shape = (shape[0], shape[2], shape[3])

    run_audit(device=args.device, input_shape=shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())

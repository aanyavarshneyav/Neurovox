#!/usr/bin/env python3
"""Environment and accelerator verification script for Neurovox.

Device-agnostic verification supporting CUDA, MPS, and CPU:
  - Selects accelerator with priority: CUDA > MPS > CPU.
  - Queries system specs, RAM, free disk space, and package versions.
  - On CUDA: reports GPU device name, driver/CUDA version, cuDNN version/flags, and bf16 support.
  - On MPS: reports MPS availability and verifies PYTORCH_ENABLE_MPS_FALLBACK=1.
  - Runs a minimal forward/backward verification pass on the selected accelerator.

MPS Fallback:
    PYTORCH_ENABLE_MPS_FALLBACK=1 enables PyTorch to route any operator not yet
    natively implemented in the Metal Performance Shaders (MPS) backend to CPU
    execution instead of raising a NotImplementedError crash.
"""

import os
import platform
import shutil
import subprocess
import sys

# Ensure MPS fallback to CPU is enabled before importing or running torch ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torchaudio


def get_hardware_info() -> dict[str, str]:
    """Retrieve system hardware specs directly from OS commands and runtime."""
    hw_info: dict[str, str] = {
        "os_version": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
    }

    # Query macOS sw_vers if available
    try:
        sw = subprocess.check_output(["sw_vers"], text=True).strip().splitlines()
        for line in sw:
            if "ProductVersion" in line:
                hw_info["macos_version"] = line.split(":")[-1].strip()
            elif "BuildVersion" in line:
                hw_info["macos_build"] = line.split(":")[-1].strip()
    except Exception:
        hw_info["macos_version"] = "N/A"

    # Query RAM via sysctl (macOS) or /proc/meminfo (Linux/Colab)
    try:
        mem_bytes = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip()
        )
        hw_info["ram_bytes"] = str(mem_bytes)
        hw_info["ram_gb"] = f"{mem_bytes / (1024**3):.2f} GB"
    except Exception:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        mem_bytes = kb * 1024
                        hw_info["ram_bytes"] = str(mem_bytes)
                        hw_info["ram_gb"] = f"{mem_bytes / (1024**3):.2f} GB"
                        break
        except Exception:
            hw_info["ram_bytes"] = "Unknown"
            hw_info["ram_gb"] = "Unknown"

    # Query free disk space
    try:
        total, used, free = shutil.disk_usage(".")
        hw_info["free_disk_gb"] = f"{free / (1024**3):.2f} GB"
    except Exception:
        hw_info["free_disk_gb"] = "Unknown"

    return hw_info


def get_accelerator_device() -> tuple[torch.device, dict[str, str]]:
    """Determine accelerator device with priority: CUDA > MPS > CPU."""
    details: dict[str, str] = {}

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        details["device_type"] = "cuda"
        details["device_name"] = torch.cuda.get_device_name(0)
        details["cuda_version"] = str(torch.version.cuda)
        details["device_count"] = str(torch.cuda.device_count())
        details["bf16_supported"] = str(torch.cuda.is_bf16_supported())
        details["cudnn_version"] = str(torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A")
        details["cudnn_benchmark"] = str(torch.backends.cudnn.benchmark)
        details["cudnn_deterministic"] = str(torch.backends.cudnn.deterministic)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        details["device_type"] = "mps"
        details["device_name"] = "Apple Silicon (MPS)"
        details["mps_built"] = str(torch.backends.mps.is_built())
        details["mps_available"] = str(torch.backends.mps.is_available())
        details["mps_fallback"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    else:
        device = torch.device("cpu")
        details["device_type"] = "cpu"
        details["device_name"] = "CPU (Fallback)"

    return device, details


def verify_forward_backward(device: torch.device) -> bool:
    """Run a tiny forward and backward pass on the active device."""
    print(f"Executing test forward/backward pass on device: {device}...")

    # Define minimal module
    linear = nn.Linear(16, 4).to(device)
    x = torch.randn(8, 16, device=device, requires_grad=True)
    target = torch.randn(8, 4, device=device)

    # Forward
    out = linear(x)
    loss = nn.functional.mse_loss(out, target)

    # Backward
    loss.backward()

    # Verify gradients
    assert linear.weight.grad is not None, "Gradient is None on linear.weight"
    assert x.grad is not None, "Gradient is None on input x"
    grad_norm = linear.weight.grad.norm().item()
    print(f"Forward/Backward successful on {device}! Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}")
    return True


def main() -> int:
    """Print environment diagnostics and run accelerator verification."""
    print("=" * 65)
    print("NEUROVOX ENVIRONMENT & ACCELERATOR VERIFICATION")
    print("=" * 65)

    hw = get_hardware_info()
    if hw.get("macos_version") != "N/A":
        print(f"macOS Version:             {hw.get('macos_version')} (Build {hw.get('macos_build', 'N/A')})")
    print(f"Platform:                  {hw['os_version']}")
    print(f"Total RAM:                 {hw['ram_gb']} ({hw['ram_bytes']} bytes)")
    print(f"Free Disk Space:           {hw['free_disk_gb']}")
    print(f"Python Version:            {hw['python_version']}")
    print(f"PyTorch Version:           {hw['torch_version']}")
    print(f"TorchAudio Version:        {hw['torchaudio_version']}")

    device, details = get_accelerator_device()
    print("-" * 65)
    print(f"Selected Device:           {device} ({details['device_name']})")
    for k, v in details.items():
        if k not in ["device_name"]:
            print(f"  {k:<24}: {v}")
    print("-" * 65)

    success = verify_forward_backward(device)
    if not success:
        print(f"FAIL: Forward/backward pass failed on {device}.")
        return 1

    print("=" * 65)
    print(f"STATUS: PASS ({details['device_name']} Ready)")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())

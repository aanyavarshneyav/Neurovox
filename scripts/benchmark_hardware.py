#!/usr/bin/env python3
"""Device-agnostic hardware throughput and thermal stability benchmark for Neurovox.

Supports CUDA (e.g. Google Colab T4), Apple Silicon MPS (Mac M3 Pro), and CPU:
  - Automatic device detection: CUDA > MPS > CPU (or override via --device).
  - Forward + backward latency, steps/sec, and peak unified/VRAM memory.
  - Candidate architectural components with synthetic tensors of real shapes:
      - Input: B=64, 1x80x180
      - Conv2d stack: C in {32, 64}
      - nn.LSTM: input=C1, hidden=C1 over T=180 for C1 in {256, 512}
      - Conv1d: in=3*d_c -> C1, k=3 at scaled widths (C=32 -> d_c=640 -> 3*d_c=1920 -> C1=256)
  - Precision: fp32 vs autocast float16 (with GradScaler on CUDA; without on MPS) vs bfloat16.
  - Records device metadata, GPU name, torch/CUDA versions, and cuDNN settings.
  - Zero hardcoded paths; supports CACHE_ROOT and DATA_ROOT environment variables.
"""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Enable MPS fallback to CPU if on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import torch.nn as nn

# Import path resolver from src.utils
try:
    from src.utils.paths import resolve_path
except ImportError:
    # Fallback if package is not installed in editable mode
    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT))
    from src.utils.paths import resolve_path


def select_device(device_arg: str | None = None) -> torch.device:
    """Select compute device with priority: CLI arg > CUDA > MPS > CPU."""
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """Synchronize GPU execution queue for accurate wall-clock timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def get_device_memory_stats(device: torch.device) -> tuple[float, float]:
    """Return (current_allocated_mb, driver_or_peak_mb)."""
    if device.type == "cuda":
        curr_mb = torch.cuda.memory_allocated(device) / (1024**2)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        return curr_mb, peak_mb
    if device.type == "mps":
        curr_mb = torch.mps.current_allocated_memory() / (1024**2)
        driver_mb = torch.mps.driver_allocated_memory() / (1024**2)
        return curr_mb, driver_mb
    return 0.0, 0.0


def empty_device_cache(device: torch.device) -> None:
    """Clear cached memory allocations."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def get_system_device_metadata(device: torch.device) -> dict[str, Any]:
    """Capture runtime device specifications and library versions."""
    meta: dict[str, Any] = {
        "device_type": device.type,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }

    if device.type == "cuda":
        meta["device_name"] = torch.cuda.get_device_name(device)
        meta["device_capability"] = torch.cuda.get_device_capability(device)
        meta["bf16_supported"] = torch.cuda.is_bf16_supported()
        meta["cudnn_version"] = (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        )
        meta["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        meta["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    elif device.type == "mps":
        meta["device_name"] = "Apple Silicon MPS"
        meta["mps_built"] = torch.backends.mps.is_built()
        meta["mps_available"] = torch.backends.mps.is_available()
        meta["mps_fallback"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    else:
        meta["device_name"] = "CPU"

    return meta


def make_conv2d_stack(c: int) -> nn.Module:
    """Instantiate plain Conv2d frontend stack (4 convs, no SKA branches, no dilations).
    
    Serves as an ablation baseline (S5-style vanilla Conv2d) and a strict lower bound.
    """
    return nn.Sequential(
        nn.Conv2d(1, c, kernel_size=3, stride=(2, 1), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(c),
        nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(c),
        nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(c),
        nn.Conv2d(c, c, kernel_size=3, stride=(2, 1), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(c),
    )


class FwSKAMock(nn.Module):
    """Frequency-wise Selective Kernel Attention mock.
    
    Contains 2 conv branches (d=1, d=2), GAP over time & channel axes, and bottleneck MLP.
    """

    def __init__(self, c: int, f: int = 40, r: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2)
        mid = max(4, f // r)
        self.mlp = nn.Sequential(
            nn.Linear(f, mid),
            nn.ReLU(),
            nn.Linear(mid, f * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u1 = self.conv1(x)
        u2 = self.conv2(x)
        u = u1 + u2
        s = u.mean(dim=(1, 3))
        attn = self.mlp(s).view(x.size(0), 2, 1, -1, 1)
        attn = torch.softmax(attn, dim=1)
        return attn[:, 0] * u1 + attn[:, 1] * u2


class CwSKAMock(nn.Module):
    """Channel-wise Selective Kernel Attention mock.
    
    Contains 2 conv branches (d=1, d=2), GAP over time & frequency axes, and bottleneck MLP.
    """

    def __init__(self, c: int, r: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2)
        mid = max(4, c // r)
        self.mlp = nn.Sequential(
            nn.Linear(c, mid),
            nn.ReLU(),
            nn.Linear(mid, c * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u1 = self.conv1(x)
        u2 = self.conv2(x)
        u = u1 + u2
        s = u.mean(dim=(2, 3))
        attn = self.mlp(s).view(x.size(0), 2, -1, 1, 1)
        attn = torch.softmax(attn, dim=1)
        return attn[:, 0] * u1 + attn[:, 1] * u2


class SE2dMock(nn.Module):
    """Squeeze-and-Excitation 2D block with reduction ratio r."""

    def __init__(self, c: int, r: int = 8) -> None:
        super().__init__()
        mid = max(4, c // r)
        self.fc = nn.Sequential(
            nn.Linear(c, mid),
            nn.ReLU(),
            nn.Linear(mid, c),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))
        w = self.fc(s).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SKAResBlockMock(nn.Module):
    """SKA Residual Block matching docs/ARCHITECTURE.md:
    Conv2d(C, C, 3, s=1, p=1) -> fwSKA -> cwSKA -> SE(r=8) + Skip.
    """

    def __init__(self, c: int, f: int = 40, r: int = 8) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c),
        )
        self.fwska = FwSKAMock(c, f=f, r=r)
        self.cwska = CwSKAMock(c, r=r)
        self.se = SE2dMock(c, r=r)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.conv(x)
        out = self.fwska(out)
        out = self.cwska(out)
        out = self.se(out)
        return self.relu(out + res)


def make_conv2d_stack_with_ska(c: int) -> nn.Module:
    """Full 2D-CNN frontend stack with 2 SKA ResBlocks.
    
    Matches docs/ARCHITECTURE.md:
      Preceding Conv2d(1, C, k=3, s=(2,1), p=1) -> ReLU -> BN2d(C)  [(B, 1, 80, T) -> (B, C, 40, T)]
      SKA ResBlock 1 (Conv2d + fwSKA + cwSKA + SE + Skip)           [(B, C, 40, T) -> (B, C, 40, T)]
      SKA ResBlock 2 (Conv2d + fwSKA + cwSKA + SE + Skip)           [(B, C, 40, T) -> (B, C, 40, T)]
      Succeeding Conv2d(C, C, k=3, s=(2,1), p=1) -> ReLU -> BN2d(C) [(B, C, 40, T) -> (B, C, 20, T)]
    """
    return nn.Sequential(
        nn.Conv2d(1, c, kernel_size=3, stride=(2, 1), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(c),
        SKAResBlockMock(c, f=40),
        SKAResBlockMock(c, f=40),
        nn.Conv2d(c, c, kernel_size=3, stride=(2, 1), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(c),
    )


class CompositeMockPipelineLowerBound(nn.Module):
    """Lower-bound composite pipeline omitting SKA branches and TDNN dilations."""

    def __init__(self, c: int = 32, c1: int = 256) -> None:
        super().__init__()
        self.c2d = make_conv2d_stack(c)
        dc = 20 * c
        self.c1d = make_conv1d_stack(3 * dc, c1)
        self.lstm = nn.LSTM(c1, c1, batch_first=True)
        self.head = nn.Linear(c1, 251)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h2 = self.c2d(x)
        h_flat = h2.view(h2.size(0), -1, h2.size(3))
        h_cat = torch.cat([h_flat, h_flat, h_flat], dim=1)
        h_tdnn = self.c1d(h_cat)
        h_lstm, _ = self.lstm(h_tdnn.transpose(1, 2))
        logits = self.head(h_lstm.mean(dim=1))
        return logits


class CompositeMockPipelineWithSKA(nn.Module):
    """Full architectural composite mock matching docs/ARCHITECTURE.md contracts:
    - 2D-CNN Frontend with 2 SKA ResBlocks (fwSKA + cwSKA + SE + Skip, dilations 1 and 2)
    - Frequency-Channel flatten to d_c = 20 * C
    - TDNN-LSTM with 3*d_c input
    - Fully-TDNN with 3 TDNN layers (dilations d=1, 2, 3)
    - MLASP attention bottleneck (3*d_i -> d_attn -> d_i) and weighted stats pooling
    - Classifier FC projection to 251 speakers
    """

    def __init__(self, c: int = 32, c1: int = 256) -> None:
        super().__init__()
        self.c2d = make_conv2d_stack_with_ska(c)
        dc = 20 * c
        self.tdnn_lstm_conv = nn.Sequential(
            nn.Conv1d(3 * dc, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(c1),
        )
        self.lstm = nn.LSTM(c1, c1, batch_first=True)
        # Fully-TDNN (3 layers, dilations 1, 2, 3)
        self.tdnn1 = nn.Sequential(
            nn.Conv1d(3 * c1, c1, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(c1),
        )
        self.tdnn2 = nn.Sequential(
            nn.Conv1d(c1, c1, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(c1),
        )
        self.tdnn3 = nn.Sequential(
            nn.Conv1d(c1, c1, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(),
            nn.BatchNorm1d(c1),
        )
        # MLASP Attention: 3*di -> d_attn -> di
        di = dc + 2 * c1
        d_attn = 128
        self.attn = nn.Sequential(
            nn.Conv1d(3 * di, d_attn, kernel_size=1),
            nn.Tanh(),
            nn.BatchNorm1d(d_attn),
            nn.Conv1d(d_attn, di, kernel_size=1),
        )
        self.bn_pool = nn.BatchNorm1d(2 * di)
        self.fc = nn.Linear(2 * di, 251)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c2d(x)
        b_sz, c_sz, f_sz, t_sz = h.shape
        xc = h.view(b_sz, c_sz * f_sz, t_sz)

        xc_cat = torch.cat([xc, xc, xc], dim=1)
        xl = self.tdnn_lstm_conv(xc_cat)
        xl_seq, _ = self.lstm(xl.transpose(1, 2))
        xl = xl_seq.transpose(1, 2)

        xl_cat = torch.cat([xl, xl, xl], dim=1)
        xt = self.tdnn1(xl_cat)
        xt = self.tdnn2(xt)
        xt = self.tdnn3(xt)

        xi = torch.cat([xc, xl, xt], dim=1)
        xi_cat = torch.cat([xi, xi, xi], dim=1)
        alpha = torch.softmax(self.attn(xi_cat), dim=-1)
        mean = (alpha * xi).sum(dim=-1)
        std = torch.sqrt(torch.clamp((alpha * xi * xi).sum(dim=-1) - mean**2, min=1e-5))
        pooled = self.bn_pool(torch.cat([mean, std], dim=1))
        return self.fc(pooled)


def make_conv1d_stack(in_channels: int, out_channels: int) -> nn.Module:
    """Instantiate Conv1d TDNN layer: Conv1d(in, out, k=3, p=1) -> ReLU -> BN1d(out)."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(out_channels),
    )


class LSTMWrapper(nn.Module):
    """Wrapper around nn.LSTM returning a single tensor for backward pass."""

    def __init__(self, c1: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=c1, hidden_size=c1, num_layers=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


def benchmark_module(
    name: str,
    module_fn: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
    precision: str,
    device: torch.device,
    target_duration_sec: float,
    warmup_steps: int = 10,
) -> dict[str, Any]:
    """Execute warm-up and timed forward+backward steps for a module."""
    result: dict[str, Any] = {
        "test_name": name,
        "precision": precision,
        "input_shape": list(input_shape),
        "status": "PASS",
        "error": None,
    }

    try:
        # Check bf16 support on CUDA
        if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            result["status"] = "SKIPPED"
            result["error"] = "bfloat16 not supported on this CUDA hardware"
            return result

        module = module_fn().to(device)
        module.train()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)

        # Precision context & scaler setup
        scaler: Any = None
        if precision == "fp32":
            autocast_ctx = contextlib.nullcontext()
        elif precision == "fp16":
            if device.type == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
                # On CUDA, use GradScaler for fp16 numerical stability
                scaler = torch.cuda.amp.GradScaler(enabled=True)
            elif device.type == "mps":
                autocast_ctx = torch.autocast(device_type="mps", dtype=torch.float16)
                scaler = None  # Prohibited on MPS per ADR-007
            else:
                raise ValueError("fp16 autocast is not supported on CPU")
        elif precision == "bf16":
            if device.type == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                scaler = None  # bf16 has full fp32 dynamic range; GradScaler not needed
            elif device.type == "mps":
                autocast_ctx = torch.autocast(device_type="mps", dtype=torch.bfloat16)
                scaler = None
            elif device.type == "cpu":
                autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
                scaler = None
            else:
                raise ValueError(f"bf16 autocast not supported on device {device}")
        else:
            raise ValueError(f"Unknown precision: {precision}")

        # Warmup loop
        for _ in range(warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(*input_shape, device=device)
            with autocast_ctx:
                out = module(x)
                loss = out.sum()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        sync_device(device)

        step_times: list[float] = []
        peak_curr_mb = 0.0
        peak_driver_mb = 0.0

        start_time = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(*input_shape, device=device)
            with autocast_ctx:
                out = module(x)
                loss = out.sum()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            sync_device(device)
            t1 = time.perf_counter()

            step_times.append(t1 - t0)
            curr_mb, driver_mb = get_device_memory_stats(device)
            if curr_mb > peak_curr_mb:
                peak_curr_mb = curr_mb
            if driver_mb > peak_driver_mb:
                peak_driver_mb = driver_mb

            elapsed = t1 - start_time
            if elapsed >= target_duration_sec and len(step_times) >= 10:
                break

        step_times_arr = np.array(step_times)
        step_times_ms = step_times_arr * 1000.0

        n = len(step_times_ms)
        early_median = float(np.median(step_times_ms[: max(1, n // 4)]))
        late_median = float(np.median(step_times_ms[max(1, 3 * n // 4) :]))
        throttling_pct = ((late_median - early_median) / early_median) * 100.0

        result.update({
            "total_steps": n,
            "total_duration_sec": round(float(step_times_arr.sum()), 2),
            "median_step_ms": round(float(np.median(step_times_ms)), 2),
            "mean_step_ms": round(float(np.mean(step_times_ms)), 2),
            "std_step_ms": round(float(np.std(step_times_ms)), 2),
            "min_step_ms": round(float(np.min(step_times_ms)), 2),
            "max_step_ms": round(float(np.max(step_times_ms)), 2),
            "steps_per_sec": round(float(1.0 / np.median(step_times_arr)), 2),
            "peak_curr_mem_mb": round(peak_curr_mb, 2),
            "peak_driver_mem_mb": round(peak_driver_mb, 2),
            "early_median_ms": round(early_median, 2),
            "late_median_ms": round(late_median, 2),
            "throttling_pct": round(throttling_pct, 2),
        })

        del module, optimizer, x, out, loss, scaler
        empty_device_cache(device)

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        empty_device_cache(device)

    return result


def main() -> int:
    """Run full benchmark suite and display formatted results."""
    parser = argparse.ArgumentParser(description="Neurovox Device-Agnostic Hardware Benchmark")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to target ('cuda', 'cuda:0', 'mps', 'cpu'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--duration-per-test",
        type=float,
        default=12.0,
        help="Target runtime per configuration in seconds (default: 12.0s, total run ~3.5 min)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test (2.0s per config) for fast verification",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="docs/benchmarks_raw.json",
        help="Path to save raw benchmark results as JSON",
    )
    args = parser.parse_args()

    duration = 2.0 if args.quick else args.duration_per_test
    device = select_device(args.device)
    device_meta = get_system_device_metadata(device)

    print("=" * 80)
    print("NEUROVOX HARDWARE BENCHMARK & THERMAL STABILITY AUDIT")
    print(f"Active Device: {device} ({device_meta['device_name']})")
    print(f"PyTorch Version: {device_meta['torch_version']}")
    if device_meta.get("cuda_version"):
        print(f"CUDA Version: {device_meta['cuda_version']} | bf16 supported: {device_meta.get('bf16_supported')}")
        print(f"cuDNN Version: {device_meta.get('cudnn_version')} | benchmark={device_meta.get('cudnn_benchmark')}, deterministic={device_meta.get('cudnn_deterministic')}")
    print(f"Target duration per test: {duration:.1f}s")
    print("=" * 80)

    B = 64
    T = 180
    F = 80

    precisions = ["fp32", "fp16", "bf16"]
    results: list[dict[str, Any]] = []

    # 1. Conv2d Stack (C=32 and C=64)
    for c in [32, 64]:
        for prec in precisions:
            print(f"--> Running Conv2d Stack (C={c}, prec={prec})...")
            res = benchmark_module(
                name=f"Conv2d_Stack_C{c}",
                module_fn=lambda c=c: make_conv2d_stack(c),
                input_shape=(B, 1, F, T),
                precision=prec,
                device=device,
                target_duration_sec=duration,
            )
            results.append(res)
            print(f"    Status: {res['status']} | Median: {res.get('median_step_ms', 'N/A')} ms | Steps/s: {res.get('steps_per_sec', 'N/A')}")

    # 1b. Conv2d Stack WITH SKA Branches (fwSKA, cwSKA, SE, dilations d=1 and d=2)
    for c in [32]:
        for prec in precisions:
            print(f"--> Running Conv2d Stack WITH SKA (C={c}, prec={prec})...")
            res = benchmark_module(
                name=f"Conv2d_Stack_with_SKA_C{c}",
                module_fn=lambda c=c: make_conv2d_stack_with_ska(c),
                input_shape=(B, 1, F, T),
                precision=prec,
                device=device,
                target_duration_sec=duration,
            )
            results.append(res)
            print(f"    Status: {res['status']} | Median: {res.get('median_step_ms', 'N/A')} ms | Steps/s: {res.get('steps_per_sec', 'N/A')}")

    # 2. nn.LSTM(C1, C1) over T=180 (C1=256 and C1=512)
    for c1 in [256, 512]:
        for prec in precisions:
            print(f"--> Running nn.LSTM (C1={c1}, prec={prec})...")
            res = benchmark_module(
                name=f"LSTM_C1_{c1}",
                module_fn=lambda c1=c1: LSTMWrapper(c1),
                input_shape=(B, T, c1),
                precision=prec,
                device=device,
                target_duration_sec=duration,
            )
            results.append(res)
            print(f"    Status: {res['status']} | Median: {res.get('median_step_ms', 'N/A')} ms | Steps/s: {res.get('steps_per_sec', 'N/A')}")

    # 3. Conv1d(3*d_c -> C1, k=3)
    conv1d_configs = [
        ("Conv1d_Scaled", 1920, 256),
        ("Conv1d_FullRef", 7680, 512),
    ]
    for c1d_name, in_ch, out_ch in conv1d_configs:
        for prec in precisions:
            print(f"--> Running {c1d_name} (in={in_ch}, out={out_ch}, prec={prec})...")
            res = benchmark_module(
                name=c1d_name,
                module_fn=lambda in_ch=in_ch, out_ch=out_ch: make_conv1d_stack(in_ch, out_ch),
                input_shape=(B, in_ch, T),
                precision=prec,
                device=device,
                target_duration_sec=duration,
            )
            results.append(res)
            print(f"    Status: {res['status']} | Median: {res.get('median_step_ms', 'N/A')} ms | Steps/s: {res.get('steps_per_sec', 'N/A')}")

    # 4. Sustained End-to-End Pipeline Lower Bound (Plain Conv2d + Conv1d + LSTM)
    print("--> Running Sustained Composite Pipeline Lower Bound (Plain Conv2d, C=32, C1=256, fp32)...")
    composite_lower_res = benchmark_module(
        name="Composite_Pipeline_LowerBound_C32_C1_256",
        module_fn=lambda: CompositeMockPipelineLowerBound(32, 256),
        input_shape=(B, 1, F, T),
        precision="fp32",
        device=device,
        target_duration_sec=duration * 2.0,
    )
    results.append(composite_lower_res)
    print(f"    Lower Bound Status: {composite_lower_res['status']} | Median: {composite_lower_res.get('median_step_ms', 'N/A')} ms | Steps/s: {composite_lower_res.get('steps_per_sec', 'N/A')}")

    # 5. Sustained End-to-End Pipeline WITH SKA Branches & Dilated Fully-TDNN
    print("--> Running Sustained Composite Pipeline WITH SKA Branches (C=32, C1=256, fp32)...")
    composite_ska_res = benchmark_module(
        name="Composite_Pipeline_with_SKA_C32_C1_256",
        module_fn=lambda: CompositeMockPipelineWithSKA(32, 256),
        input_shape=(B, 1, F, T),
        precision="fp32",
        device=device,
        target_duration_sec=duration * 2.0,
    )
    results.append(composite_ska_res)
    print(f"    Full SKA Status: {composite_ska_res['status']} | Median: {composite_ska_res.get('median_step_ms', 'N/A')} ms | Steps/s: {composite_ska_res.get('steps_per_sec', 'N/A')} | Throttling: {composite_ska_res.get('throttling_pct', 'N/A')}%")

    print("\n" + "=" * 105)
    print(f"{'Module Name':<32} {'Prec':<6} {'Median(ms)':<12} {'Steps/s':<10} {'Peak Cur(MB)':<14} {'Peak Drv(MB)':<14} {'Throttle %':<10}")
    print("-" * 105)
    for r in results:
        if r["status"] == "PASS":
            print(
                f"{r['test_name']:<32} {r['precision']:<6} "
                f"{r['median_step_ms']:<12.2f} {r['steps_per_sec']:<10.2f} "
                f"{r['peak_curr_mem_mb']:<14.2f} {r['peak_driver_mem_mb']:<14.2f} "
                f"{r['throttling_pct']:<10.2f}%"
            )
        else:
            print(f"{r['test_name']:<32} {r['precision']:<6} {r['status']}: {r['error']}")
    print("=" * 105)

    payload = {
        "metadata": device_meta,
        "results": results,
    }

    output_path = resolve_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Benchmark results successfully written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

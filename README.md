# Neurovox

> **Scaled Reproduction of *"A Hybrid Neural Approach to Speaker Verification with an Improved Additive Angular Margin Loss"***  
> Authors: Jahangir Alam, Abderrahim Fathan, and Md Shahidul Alam (IEEE IJCNN 2025)  
> Target Corpus: LibriSpeech ASR Corpus (OpenSLR 12)  
> Hardware Target: Apple Mac M3 Pro (MPS / CPU) & Google Colab (Secondary)

---

## 1. Project Overview

Neurovox is a rigorous, scientific reproduction of the **SKA-HNN** speaker embedding backbone and **IAAMSoftmax** (Improved Additive Angular Margin Softmax with AdaptiveDrop label cleansing) loss function.

This project scales channel parameters ($C=32, C_1=256$) to train robustly on LibriSpeech `train-clean-100` (251 speakers) on local Mac M3 Pro hardware while evaluating parameter-matched baselines (ECAPA-TDNN) and isolating the anti-noise mechanics of AdaptiveDrop.

---

## 2. Documentation Index

- [AGENTS.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/AGENTS.md): Standing operational rules, scientific integrity principles, and non-negotiables.
- [docs/PROJECT_SPEC.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/PROJECT_SPEC.md): Corrected, consolidated specification with provenance tags and PRD audit trail.
- [docs/ARCHITECTURE.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/ARCHITECTURE.md): Module-by-module contracts, tensor shapes, and S0–S6 ablation mappings.
- [docs/DATA_PROTOCOL.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/DATA_PROTOCOL.md): Partitions, 100% cross-chapter target trial generation, and Mac M3 Pro I/O plan.
- [docs/EXPERIMENT_PLAN.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/EXPERIMENT_PLAN.md): Five-tier roadmap, technical gates, formal hypotheses, and statistical protocols.
- [docs/DECISIONS.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/DECISIONS.md): Architectural Decision Records (ADRs).
- [docs/OPEN_QUESTIONS.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/OPEN_QUESTIONS.md): Registry of unverified items, paper ambiguities, and working assumptions.
- [docs/RESULTS_LOG.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/RESULTS_LOG.md): Append-only experimental ledger recording measured runs.

---

## 3. Environment Setup

### Local Mac M3 Pro Setup
```bash
# 1. Clone repository and navigate to root
git clone https://github.com/aanyavarshneyav/Neurovox.git
cd Neurovox

# 2. Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# 4. Install dependencies in editable mode
pip install -e ".[dev,demo]"

# 5. Verify installation with test suite
pytest -v
```

---

## 4. Phase 1 Task 1: Hardware Benchmark

Before launching training runs, Phase 1 commences with the hardware benchmark task:
```bash
python scripts/benchmark_hardware.py --device auto
```
This benchmarks FLAC decoding throughput, step latency on `mps`, and peak memory, and runs `torchinfo` to verify exact parameter counts before locking compute budgets.

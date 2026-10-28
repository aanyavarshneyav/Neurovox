---
description: Rules for config-driven experiment execution, seeding, and results logging
alwaysApply: false
---

# Experiment Execution & Logging Rules

1. **Strictly Config-Driven Execution:**
   - Every experiment must be fully defined by a version-controlled YAML configuration file in `configs/`.
   - No hardcoded hyperparameters in scripts; scripts must accept `--config <path>` and parse arguments via OmegaConf or argparse.

2. **Random Seed Discipline:**
   - Every run must explicitly set and log its random seed.
   - Deterministic seed initialization must cover Python `random`, `numpy.random`, and `torch.manual_seed` (and `torch.cuda.manual_seed_all` if CUDA).
   - Headline comparisons (e.g. S0 vs S1, noise sweeps) must run across at least 3 distinct seeds (`42`, `43`, `44`).

3. **Atomic Result Logging:**
   - Upon completion of any evaluation run, an entry must be appended to `docs/RESULTS_LOG.md` recording:
     - `Run ID`: standardized format matching `<tier>_<model>_<loss>_<dataset>_<condition>_s<seed>`.
     - `Config Hash`: SHA-256 hash of the configuration file.
     - `Seed`: integer random seed.
     - `Measured Params`: from `torchinfo.summary()`.
     - `EER (%)`: equal error rate.
     - `minDCF ($P=0.01$)` and `minDCF ($P=0.05$)`: dual priors with $C_{miss}=1.0, C_{fa}=1.0$.
     - `Device`: `mps`, `cuda`, or `cpu`.
     - `Notes`: factual observations only (no unmeasured claims).

4. **Checkpoint Management:**
   - Save checkpoints to `checkpoints/<run_id>/`:
     - `best_eer.pt`: model state dict with lowest validation EER.
     - `latest.pt`: current state dict and optimizer state for resumption.
     - `config.yaml`: copy of the exact configuration used.
   - Checkpoints are git-ignored and must never be committed to repository history.

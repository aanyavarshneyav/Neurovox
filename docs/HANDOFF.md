# Session Handoff & State Summary

**Date:** 2026-11-03  
**Current Phase:** Phase 2A Complete (Pipeline Architecture, Trainer, Extractor, Loss, and Sanity Runs A/B/C Verified)  
**Git Head:** `origin/phase-2` (uncommitted Phase 2A changes pending explicit user permission)  
**Test Status:** All tests passing (71 passed in 5.04s)

---

## 1. Accomplishments This Session

1. **Lightweight Baseline Architecture (`src/models/small_tdnn_se.py`):**
   - Implemented `small_tdnn_se` (1D Dilated Conv + Squeeze-and-Excitation + Statistics Pooling with variance clamp $\ge 10^{-5}$ + Linear projection to 192-dim).
   - Parameter audit executed via `torchinfo.summary()`: strictly measured at **410,928 parameters (0.411M)**.
   - Verified layer shapes and forward pass across arbitrary batch lengths in `tests/test_small_tdnn_se.py` (3 passed).

2. **AAMSoftmax Loss with fp32 Numerics & Monotonic Margin Fallback (`src/losses/aam_softmax.py`):**
   - Margin $m=0.2$ and scale $s=30.0$, strictly applying angular margin exclusively to target class $j = y_i$ and normalizing denominator over all $j$.
   - Executes in `torch.float32`. Incorporates monotonic linear fallback when $\theta + m > \pi$ to prevent gradient inversion.
   - Comprehensive unit tests in `tests/test_losses.py` (5 passed): $s=1, m=0$ with random unit-norm vectors yields loss $\approx \ln(251) \pm 0.05$; $s=30, m=0.2$ matches 64-bit NumPy reference implementation within $1\times 10^{-5}$; finite non-zero gradients verified across multiple seeds.

3. **Full-Utterance Embedding Extractor (`src/eval/extractor.py`):**
   - Implemented `EmbeddingExtractor` processing full audio utterances (zero temporal cropping) in `model.eval()` mode with instance Cepstral Mean Normalization (CMN).
   - Enforces default batch size 1 and length-bucketed batch extraction, mathematically guaranteeing zero-padding batch invariance (verified within $< 1\times 10^{-4}$ in `tests/test_extractor.py`).
   - Parses trial lists directly to extract only participating utterances, maintaining an in-memory cache to eliminate redundant audio decoding.

4. **Modular Trainer with Bitwise Stateful Resume & NaN Interceptors (`src/training/trainer.py`, `scripts/train.py`):**
   - Adam optimizer with triangular CyclicLR schedule (`cycle_momentum=False`), configurable `grad_clip=3.0`, and 4-stage per-step NaN/Inf detection guards.
   - Stateful checkpointing saves model, loss, optimizer, scheduler, and all 4 system RNG states (`torch`, `numpy`, `python`, `mps`).
   - Strict resume parity unit-tested in `tests/test_trainer.py`: $< 1\times 10^{-5}$ on CPU and $< 1\times 10^{-4}$ on MPS between uninterrupted runs and interrupted-then-resumed runs.
   - Exports per-epoch training and evaluation metrics separately to JSON and CSV artifacts.

5. **Train-Probe Fixed-Seed Trial Generation (`data/trials/trials_train-probe.txt`):**
   - Generated 5,852 cross-chapter, gender-matched trial pairs from `train-clean-100` speakers to monitor training-data EER alongside unseen `dev-clean` EER every epoch without touching test partitions.

6. **End-to-End Pipeline Sanity Runs (All Gates Passed):**
   - **Sanity Run A (`sanity_2a_overfit`):** Overfit 1 fixed batch of 64 speakers, constant LR $1\times 10^{-3}$, 300 steps, augmentations completely OFF. Loss reduced from $13.3750 \to 0.000011$ ($1,207,242\times$ reduction, threshold $\ge 100\times$), $100.00\%$ train accuracy, zero NaNs.
   - **Sanity Run B (`sanity_2a_50spk`):** 50-speaker subset, $B=32$ (deliberate ADR-016 deviation), 3 epochs, 178 batches/epoch, cyclic LR, augmentations ON, RIR OFF. Loss decreased monotonically ($8.7093 \to 5.1842 \to 3.6097$), train accuracy increased ($26.0\% \to 65.2\% \to 78.1\%$), out-of-domain dev EER reached 20.58%, zero NaNs.
   - **Sanity Run C (`sanity_2a_full`):** Full 251 speakers, $B=64$, 446 batches/epoch, 10 epochs (4,460 steps), cyclic LR ($1\times 10^{-7} \to 1\times 10^{-3}$), speed/noise/SpecAugment ON, RIR OFF. Loss decreased $10.5741 \to 2.5086$, train accuracy reached $88.50\%$. `train-probe` EER reached $3.90\%$, `dev-clean` EER reached $10.71\%$ (minDCF 0.4539). The train-vs-dev generalization gap remained strictly bounded between $+5.95\%$ and $+8.39\%$ throughout training.

7. **Documentation, Decisions & Scientific Integrity:**
   - Recorded **ADR-020** in `docs/DECISIONS.md`.
   - Logged all 3 sanity runs in `docs/RESULTS_LOG.md` with explicit config hashes, measured metrics, seeds, hardware (`Mac M3 Pro, mps, fp32`), and active augmentations.
   - Total test count expanded from 56 to 71 passing tests.

### Deterministic Data Subset & Trial Generation for Colab / Fresh VMs
On Google Colab or fresh VMs where `.gitignore` excludes `data/`:
1. **50-Speaker Subset Manifest (`data/manifests/train-50spk.csv`):**
   Run `python3 scripts/make_subset_manifest.py --source data/manifests/train-clean-100.csv --output data/manifests/train-50spk.csv --num-speakers 50`.
   This deterministically sorts unique speaker IDs and maps labels 0..49, guaranteeing SHA-256: `f63c4dc99eb41b45880fcf527aa724160e6b4860390cc7c0221a06a27f5f46da`.
2. **Train-Probe Trial List (`data/trials/trials_train-probe.txt`):**
   Generated deterministically via `scripts/make_train_probe_trials.py` (fixed seed 42, 5,852 trials, cross-chapter targets, gender-balanced), guaranteeing SHA-256: `381b6f3807221ff69e8e2fa51571d79e6cb3dfc46002f5e3ba0ee5df2d56a2bb`.
3. **Verification via `scripts/verify_trials.py`:**
   Running `python3 scripts/verify_trials.py` verifies all 8 committed trial lists and the subset manifest against their authoritative SHA-256 hashes, raising an error if any deviate.

---

## 2. Next 3 Immediate Tasks

1. **Phase 2B: Core SKA-HNN Architecture Implementation:**
   - Implement `src/models/resblock_ska.py` (2D-CNN stem, fwSKA frequency-wise attention, cwSKA channel-wise attention, SE block, residual skip).
   - Implement `src/models/tdnn_lstm.py` (TDNN layers, BiLSTM / LSTM temporal acoustic modelling).
   - Implement `src/models/mlasp.py` (Multi-Level Attentive Statistics Pooling with local statistics window $\mu_t, \sigma_t$ clamped $\ge 10^{-5}$ in fp32, and MLFC highway).
   - Assemble full composite `src/models/ska_hnn.py` with ablation switches S0–S6.
2. **Phase 2B Parameter Audit & Shape Contract Verification:**
   - Run `.agents/workflows/param-audit.md` with `torchinfo.summary()` on scaled ($C=32, C_1=256$) and full ($C=128, C_1=512$) SKA-HNN variants.
   - Record measured parameter counts in `docs/PROJECT_SPEC.md` and `docs/RESULTS_LOG.md`.
   - Write comprehensive unit tests for all ablation switches S0–S6 verifying output tensor shapes and non-zero gradients.
3. **Phase 2B Loss: IAAMSoftmax with AdaptiveDrop:**
   - Implement `src/losses/iaam_softmax.py` featuring AdaptiveDrop sample filtering ($\tau = 0.423$) and label correction ($\tau_{\text{corr}} = 0.5$) with warm-up schedules ($ESD=5, ESC=5$) and minimum retention safety guard (`min_keep_ratio=0.5`).
   - Verify numerically against unit tests across seeds.

---

## 3. Active Risks & Notes for Next Agent

- **Test Partition Protection:** Strict firewall remains in effect. `test-clean` and `test-other` were NOT accessed during Phase 2A and must NOT be accessed until final frozen evaluation. All validation and model selection uses `dev-clean`.
- **MPS Fallback Guard:** Always run with `PYTORCH_ENABLE_MPS_FALLBACK=1` on Mac M3 Pro.
- **Git Commit Rule:** No commits can be executed without explicit user permission. A commit proposal should be presented to the user at the conclusion of this turn.

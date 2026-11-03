# Session Handoff & State Summary

**Date:** 2026-11-04  
**Current Phase:** Phase 2B Complete (SKA-HNN Model Family, S0–S6 Ablation Switches, Measured Parameter Audit, Overfit & Smoke Runs Passed)  
**Git Head:** `origin/phase-2`  
**Test Status:** All tests passing (92 passed in 13.47s)

---

## 1. Accomplishments This Session (Phase 2B Complete)

1. **Modular SKA-HNN Architecture (`src/models/ska_hnn/`):**
   - Implemented full modular suite: `local_stats.py` (fp32 sliding-window $W=21$ with replicate padding and variance clamp $\ge 10^{-5}$), `fw_ska.py` (frequency-wise SKA with 2 branches, $r=8$, softmax over branch `dim=1`), `cw_ska.py` (channel-wise SKA with 2 branches, $r=8$, softmax over branch `dim=1`), `se.py` (2D Squeeze-and-Excitation), `resblock.py` (`SKAResBlock` with toggle for S5 vanilla conv), `tdnn_lstm.py` (TDNN + unidirectional LSTM + ReLU + BN1d), `fully_tdnn.py` (3 cascaded dilated Conv1d layers with $d=1, 2, 3$), `mlfc.py` (Multi-Level Feature Combination $[X_c; X_l; X_t]$), `mlasp.py` (Multi-Level Attentive Statistics Pooling with channel-dependent attention, softmax over temporal `dim=2`, fp32 weighted mean+std pooling), and `ska_hnn.py` assembling the complete backbone.
   - Preserves strict `extract_layer="auto"` contract: returns primary embedding FC1 ($2d_i \to 192$, closest to pooling layer per paper Sec. II-C) in `eval()` mode, and FC2 in `train()` mode.

2. **Ablation Switches S0–S6:**
   - Builder function `build_ska_hnn` and config constructor `build_ska_hnn_from_config` support switches S0 through S6.
   - S0 and S1 share identical backbone architecture (4,167,700 parameters); the distinction is exclusively in the training loss (IAAMSoftmax vs standard AAMSoftmax).
   - Parameter deltas strictly verified in expected directions: S0 = S1 > S5 > S2 > S3 > S4 > S6.

3. **Measured Parameter Audit (`torchinfo`):**
   - Measured via `torchinfo.summary()` at scaled dimensions ($C=32, C_1=256, d_{attn}=128, \text{emb}=192$):
     - S0 / S1 (Full Proposed / AAMSoftmax): **4,167,700** parameters
     - S2 (No MLASP Context): **3,872,788** parameters
     - S3 (No MLFC): **3,360,404** parameters
     - S4 (No MLFC & No Context): **3,294,868** parameters
     - S5 (Vanilla 2D-CNN): **4,109,728** parameters
     - S6 (No 2D-CNN Frontend): **2,195,584** parameters
   - Documented discrepancy against PRD's hand-calculated 2.65M estimate (caused by reverse-engineering $d_{attn}=614$ in PRD) and paper Table II S1 $\to$ S5 13.54M delta (tracked in `OPEN_QUESTIONS.md` Q2).

4. **Rigorous Validation & Extractor Invariance:**
   - Expanded unit tests in `tests/test_ska_hnn.py` to 9 tests covering variable $T \in \{100, 180, 400\}$, attention weight summation to 1.0 along named dimensions, fp32 local stats without NaN on constant input, full backward grad reach across all parameters, eval-mode batch invariance, and switch parameter ordering.
   - Verified `EmbeddingExtractor` with `SKAHNN`: bucketed extraction of variable-length waveforms strictly matches single-waveform unpadded extraction within $< 4\times 10^{-7}$ (threshold $< 1\times 10^{-4}$). Tested that naive zero-padding distorts embeddings ($\Delta \approx 0.114$), verifying the extractor's length-bucketed/alone evaluation protocol.
   - Repository test suite stands at **92 passing tests**.

5. **Completed Smoke & Overfit Runs (Gates Passed):**
   - `sanity_2b_overfit_s0`: Initial loss 13.6175 $\to$ Final 0.000012, 100.0% acc, $1,095,755\times$ loss reduction, 0 NaNs.
   - `sanity_2b_overfit_s1`: Initial loss 13.6175 $\to$ Final 0.000012, 100.0% acc, $1,095,755\times$ loss reduction, 0 NaNs.
   - `sanity_2b_50spk_s0`: 2 epochs, B=32, cyclic LR. Step-0 loss 11.6978 (dev EER 33.06%) $\to$ Ep 1 loss 8.2440 (dev EER 29.84%) $\to$ Ep 2 loss 3.6219 (dev EER 20.81%). Model forward+backward compute accounts for ~96% of total step time (268.2 ms pure model vs 278.7 ms total step time).

---

## 2. Next Immediate Step: Phase 2C (Loss Functions & Baseline Alignment)

1. **IAAMSoftmax with AdaptiveDrop (`src/losses/iaam_softmax.py`):**
   - Implement online sample dropping with threshold $\tau = 0.423$, drop start epoch $ESD=5$, and sample drop safety guard (`min_keep_ratio=0.5`).
   - Implement label correction before sample filtering with threshold $\tau_{\text{corr}} = 0.5$ and start epoch $ESC=5$.
2. **AAMSoftmax Comparison Reference:**
   - Reuse existing verified `src/losses/aam_softmax.py` for Switch S1 comparison runs.
3. **Parameter-Matched ECAPA-TDNN Baseline (`src/models/ecapa_tdnn.py`):**
   - Size channel dimension $C$ to match measured SKA-HNN parameter count (~4.17M) and verify with `torchinfo`.

---

## 3. Active Risks & Operational Guards

- **Test Partition Protection:** Strict firewall remains in effect. `test-clean` and `test-other` must NOT be accessed until final frozen evaluation. All validation and model selection uses `dev-clean`.
- **MPS Fallback Guard:** Always run with `PYTORCH_ENABLE_MPS_FALLBACK=1` on Mac M3 Pro.
- **Git Commit Rule:** No commits can be executed without explicit user permission.

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
   Generated deterministically via `scripts/make_train_probe_trials.py` (fixed seed 42, 5,852 trials, cross-chapter targets, gender-balanced), guaranteeing SHA-256: `381b6f38c398d487a141305a9c3d10937364d85016f3d07db97cacb85840d8c2`.
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

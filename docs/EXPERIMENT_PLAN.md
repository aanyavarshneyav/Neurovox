# EXPERIMENT_PLAN.md — Tiers, Hypotheses & Technical Gates

**Project:** Scaled Reproduction of SKA-HNN and IAAMSoftmax on LibriSpeech  
**Compute Execution:** Local Apple Mac M3 Pro (MPS primary with CPU fallback, `PYTORCH_ENABLE_MPS_FALLBACK=1`). No Colab.  
**Policy:** Course has no AI-use policy; AI-written code is permitted across all components (ADR-012).  
**Phase 1 Kickoff Task:** Hardware Benchmark & Environment Verification (Task 1).

---

## 1. Experimental Roadmap & Tier Hierarchy

To ensure incremental progress with continuous validation, experimental execution is structured into five sequential tiers (Tier 0 through Tier 4). All empirical expectations are framed as scientific hypotheses, not predetermined facts.

```
[Phase 1 Benchmark: Hardware Throughput & Parameter Audit]
                           │
                           ▼
[Tier 0: Baselines] ─── Parameter-matched ECAPA-TDNN & Standard ECAPA-TDNN
                           │
                           ▼
[Tier 1: Core Systems] ─ Scaled SKA-HNN S0 (IAAMSoftmax) & S1 (AAMSoftmax)
                           │
                           ▼
[Tier 2: Ablations] ──── Systems S2, S3, S4, S5, S6 (Inductive Bias Study)
                           │
                           ▼
[Tier 3: Label Noise] ── Part A: Synthetic Noise (10%, 20%, 40%)
                         Part B: WavLM-Base+ K-Means Pseudo-Labels
                           │
                           ▼
[Tier 4: Extensions] ─── AS-Norm & Zero-Shot VoxCeleb1-O Evaluation
```

---

## 2. Phase 1 Mandatory First Task: Hardware Benchmark

Before launching full multi-epoch training runs, Task 1 of Phase 1 will execute an empirical hardware benchmark on the local Mac M3 Pro:

- **Benchmark Script:** `scripts/benchmark_hardware.py`
- **Measured Variables:**
  - Raw audio loading and FLAC decoding throughput (samples/sec).
  - 80-dim log-Mel filterbank extraction latency per 1.8s audio chunk.
  - Forward + backward pass step latency (ms/step) on `mps` vs `cpu` for batch size $B=64$.
  - Peak unified memory consumption (GB) across training steps.
  - Parameter audit via `torchinfo` for all candidate models.
- **Outcome:** The results will be logged in `docs/RESULTS_LOG.md` (Run ID: `bench_m3pro_01`), and total epoch budgets and batch sizes will be formally calibrated based on real measured throughput.

---

## 3. Tier Specifications, Hypotheses, and Technical Gates

### Tier 0: Baselines (ECAPA-TDNN)
- **Objective:** Establish clean, reproducible end-to-end baseline pipelines on LibriSpeech using ECAPA-TDNN.
- **Models:**
  1. Parameter-matched ECAPA-TDNN: width chosen so its measured torchinfo count is close to the measured scaled SKA-HNN count (width and count TBD: measure with torchinfo; this requires the SKA-HNN to be built and audited first) `[Engineering Choice]`.
  2. Standard-width ECAPA-TDNN ($C=1024$; count TBD: measure with torchinfo; the ~14.7M figure from the PRD/AGENTS is `[External-UNVERIFIED]`) `[Engineering Choice]`.
- **Loss:** Standard AAMSoftmax ($s=30, m=0.2$).
- **Formal Hypothesis (H0):** *The parameter-matched ECAPA-TDNN baseline will achieve stable convergence on LibriSpeech without loss divergence or numerical collapse, establishing an unbiased capacity-controlled reference.*
- **Technical Gates (Pass Criteria):**
  1. Training runs for designated epochs without NaN/Inf loss or gradient explosion.
  2. Output embeddings have unit norm and non-collapsed pairwise cosine distribution.
  3. Metric calculation unit tests pass against reference implementations (`speechbrain.utils.metric_stats`).
  4. Trial generation generates 100% cross-chapter target trials.

---

### Tier 1: Core Proposed Systems (SKA-HNN S0 & S1)
- **Objective:** Train the core scaled SKA-HNN architecture under IAAMSoftmax (S0) and AAMSoftmax (S1).
- **Architecture:** Scaled SKA-HNN ($C=32, C_1=256$).
- **Evaluations:** Embeddings from FC1 (closest to pooling, per paper `[Paper]`) are the primary result. FC2 is scored as a logged diagnostic `[Engineering Choice]`.
- **Seeds:** 3 seeds (42, 43, 44) for headline comparison.
- **Formal Hypothesis (H1) [HYPOTHESIS]:** On clean LibriSpeech, S0 and S1 will differ little, since the paper reports only a small supervised-regime gap (Table IV). This may not hold at 251 speakers with 3 seeds; either outcome, or a reversed ordering, will be reported as measured.
- **Technical Gates (Pass Criteria):**
  1. Layer output shapes match `docs/ARCHITECTURE.md` contracts (unit-tested).
  2. `torchinfo` parameter audit runs and counts are logged in RESULTS_LOG.
  3. Gradient norms are finite and non-zero for every module (2D-CNN, LSTM, TDNN, attention).
  4. No NaN/Inf in loss or embeddings; per-step AdaptiveDrop keep-ratio and guard-trigger counts are *logged* (no threshold on their values).

---

### Tier 2: Ablation Suite (S2, S3, S4, S5, S6)
- **Objective:** Systematically isolate the empirical contributions of MLASP context, MLFC highway, SKA attention, and the 2D-CNN frontend.
- **Ablation Systems:**
  - **S2:** S1 without MLASP local statistics context ($X_i$ input).
  - **S3:** S1 without MLFC ($X_i = X_t$).
  - **S4:** S1 without MLFC and without context ($X_i = X_t$, no local stats).
  - **S5:** S1 with vanilla 2D-CNN (single-branch $3\times 3$ convolutions without SKA).
  - **S6:** S1 without 2D-CNN (MFB features fed directly to TDNN-LSTM).
- **Formal Hypothesis (H2) [HYPOTHESIS]:** The paper's Table II ordering (S0 best; S6 worst) may or may not reproduce at 251 speakers. Single-seed differences may be non-monotonic. Results are reported as measured, with 3 seeds and bootstrap CIs, and non-monotonic results are not retried.
- **Technical Gates (Pass Criteria):**
  1. Each ablation system instantiates and runs through train/val loops without runtime error.
  2. Parameter counts for each variant are measured via `torchinfo` and logged side-by-side with metrics.
  3. All results (monotonic or non-monotonic) are logged faithfully in `docs/RESULTS_LOG.md`.

---

### Tier 3: Label-Noise Robustness & Self-Supervised Pseudo-Labels
- **Objective:** Validate the primary scientific contribution of IAAMSoftmax/AdaptiveDrop: mitigating label noise memorization.
- **Part A (Controlled Synthetic Noise — Primary Scientific Focus):**
  - Uniform symmetric label corruption applied to clean training labels at rates: $10\%$, $20\%$, and $40\%$.
  - Models: S0 (IAAMSoftmax) vs. S1 (AAMSoftmax) across 3 seeds.
- **Part B (Realistic SSL Proxy):**
  - Extract frozen self-supervised representations using `WavLM-Base+` (without speaker labels).
  - Perform K-Means clustering into $K=251$ pseudo-speaker clusters.
  - Log cluster purity and NMI against ground truth speaker IDs.
  - Train S0 vs. S1 on pseudo-labeled data.
- **Formal Hypothesis (H3) [HYPOTHESIS]:** IAAMSoftmax degrades less than AAMSoftmax as label noise grows. **Null outcome is explicitly possible:** a small model may not memorize noisy labels within the epoch budget (PRD_review issue 4), in which case no gap appears. We log noisy-label training accuracy per epoch to check whether memorization occurred, and report the result as measured.
- **Technical Gates (Pass Criteria):**
  1. Label noise injector unit tests verify exact noise rates ($10\%, 20\%, 40\%$).
  2. No supervised speaker labels leak into the WavLM feature extraction or clustering pipeline.
  3. AdaptiveDrop runs without error when all/none of a batch is filtered (guard path unit-tested; `min_keep_ratio` is a switch and the guard can be disabled for an ablation).
  4. Validation EER, keep-ratio, guard-trigger count and noisy-label train accuracy are logged per epoch.
  5. Cluster purity and NMI of pseudo-labels against true speakers are computed and logged; K=251 uses the true speaker count (as the paper does with 5,994), so also log K sensitivity.

---

### Tier 4: Extensions & Score Calibration
- **Objective:** Explore score normalization and zero-shot out-of-domain evaluation.
- **Components:**
  1. Adaptive Symmetric Score Normalization (AS-Norm) using an unlabelled cohort from `dev-clean`.
  2. Zero-shot evaluation on official `VoxCeleb1-O` verification trial list (inference-only).
- **Formal Hypothesis (H4) [HYPOTHESIS]:** AS-Norm changes minDCF on out-of-domain trials; direction is not assumed. VoxCeleb1-O needs registration/download over 1 GB, which needs your confirmation first.
- **Technical Gates (Pass Criteria):**
  1. AS-Norm calibration runs deterministically on trial score files.
  2. Zero-shot pipeline outputs valid EER and minDCF on VoxCeleb1-O trials.

---

## 3b. Calibrated Compute Budget & Feasibility

Following Phase 1A measurements logged in [docs/BENCHMARKS.md](BENCHMARKS.md):
- **Batch compute latency:**
  - *Full Model with SKA:* **$411.52\text{ ms/batch}$** ($B=64$, Composite Scaled SKA-HNN on Mac M3 Pro, `fp32`) `[Measured]`.
  - *Lower Bound (Plain Conv2d):* **$116.51\text{ ms/batch}$** (omitting SKA branches and TDNN dilations) `[Measured]`.
- **DataLoader batch latency:** **$26.36\text{ ms/batch}$** ($4\text{ workers}$ on genuine LibriSpeech FLAC) `[Measured]`. Data loading latency is completely hidden by asynchronous pre-fetching.
- **Batches per 28,539-utterance epoch:** $\lceil 28,539 / 64 \rceil = \mathbf{446\text{ batches}}$ `[Derived]`.
- **Estimated time per epoch (Full SKA):** $446 \times 0.41152\text{ s} \times 1.10 \approx \mathbf{201.89\text{ seconds}}$ ($\approx \mathbf{3.36\text{ minutes/epoch}}$) including validation and logging `[ESTIMATE]`.
- **Estimated time per 30-epoch run (Full SKA):** $30 \times 201.89\text{ s} \approx \mathbf{100.9\text{ minutes}}$ ($\approx \mathbf{1.68\text{ hours}}$) `[ESTIMATE]`.
- **Daily capacity (Mac M3 Pro):** $24\text{ hours} / 1.68\text{ hours/run} \approx \mathbf{14.3\text{ full 30-epoch runs per 24 hours}}$ `[ESTIMATE]`.

---

## 3c. Run Location Decision & Workload Allocation

All experimental runs execute locally on Apple Silicon Mac M3 Pro (`device = mps`, `torch.float32`). Google Colab (Tesla T4) is not pursued; Colab configuration and runner (`notebooks/colab_runner.ipynb`) remain available solely as a secondary fallback if needed.

### Plan-Level Compute Budget & Proposed Allocation

The roadmap comprises **51 total runs** across 5 tiers:
- **Tier 0 Baselines:** 2 models (matched, standard) $\times$ 3 seeds $= 6\text{ runs}$ ($\approx 10.1\text{ hours}$).
- **Tier 1 Core Systems:** 2 models (S0, S1) $\times$ 3 seeds $= 6\text{ runs}$ ($\approx 10.1\text{ hours}$).
- **Tier 2 Ablations:** 5 systems (S2–S6) $\times$ 3 seeds $= 15\text{ runs}$ ($\approx 25.3\text{ hours}$).
- **Tier 3 Part A (Synthetic Noise):** 3 noise levels ($10\%, 20\%, 40\%$) $\times$ 2 models $\times$ 3 seeds $= 18\text{ runs}$ ($\approx 30.3\text{ hours}$).
- **Tier 3 Part B (WavLM Pseudo-Labels):** 2 models $\times$ 3 seeds $= 6\text{ runs}$ ($\approx 10.1\text{ hours}$).
- **Tier 4 Extensions:** AS-Norm and VoxCeleb1-O zero-shot (inference-only on existing checkpoints, $0\text{ training runs}$).

**Plan-Level Total Estimates:**
- **Full Portfolio (51 runs = 1,530 epochs):**
  $$\text{Total Compute Time} = 1,530\text{ epochs} \times 201.89\text{ s/epoch} \approx \mathbf{308,892\text{ seconds}} \approx \mathbf{85.8\text{ hours}}\quad (\approx \mathbf{3.58\text{ days}})\quad \text{[ESTIMATE]}$$
- **Pruned Core Portfolio (31 runs = 930 epochs):**
  $$\text{Total Compute Time} = 930\text{ epochs} \times 201.89\text{ s/epoch} \approx \mathbf{187,758\text{ seconds}} \approx \mathbf{52.2\text{ hours}}\quad (\approx \mathbf{2.17\text{ days}})\quad \text{[ESTIMATE]}$$

> [!IMPORTANT]
> **Re-measurement Requirement:** These plan-level estimates are computed from synthetic benchmark step latencies and must be formally re-measured and calibrated after the first real training epoch executes.

**Proposed Cut Order (if constrained by local execution time):**
1. **Cut Priority 1 (Noise sweep pruning):** Drop the $10\%$ synthetic noise sweep in Tier 3 Part A, retaining $20\%$ and $40\%$ noise. *Saves 6 runs ($\approx 10.1\text{ hours}$).*
2. **Cut Priority 2 (Ablation seed reduction):** Execute Tier 2 ablations S2–S6 on single seed (`seed 42`) first, running additional seeds only for variants where performance differences are within error margins. *Saves 10 runs ($\approx 16.8\text{ hours}$).*
3. **Cut Priority 3 (SSL seed reduction):** Execute Tier 3 Part B (WavLM pseudo-labels) on single seed (`seed 42`). *Saves 4 runs ($\approx 6.7\text{ hours}$).*

*Pruned Core Portfolio:* **31 runs** ($\approx 52.2\text{ hours}$ total). User to select preferred allocation.

---

## 4. Run Naming & Configuration Convention

All experimental runs follow an explicit, parseable naming scheme:

```
<tier>_<model>_<loss>_<dataset>_<condition>_s<seed>
```

**Examples:**
- `t0_ecapa_matched_aam_libri_clean_s42` (Tier 0 parameter-matched baseline)
- `t0_ecapa_std_aam_libri_clean_s42` (Tier 0 standard SpeechBrain baseline)
- `t1_skahnn_s0_iaam_libri_clean_s42` (Tier 1 proposed S0 system, seed 42)
- `t1_skahnn_s1_aam_libri_clean_s42` (Tier 1 baseline S1 system, seed 42)
- `t2_skahnn_s2_aam_libri_clean_s42` (Tier 2 ablation S2)
- `t3_skahnn_s0_iaam_libri_noise20_s42` (Tier 3 S0 under 20% synthetic noise)
- `t3_skahnn_s1_aam_libri_noise20_s42` (Tier 3 S1 under 20% synthetic noise)
- `t3_skahnn_s0_iaam_libri_ssl_wavlm_s42` (Tier 3 S0 on WavLM pseudo-labels)

Every run must correspond to a distinct YAML configuration file in `configs/` and an atomic entry in `docs/RESULTS_LOG.md`.

---

## 5. Statistical Rigor Protocol

1. **Seeds:** Random seeds `42`, `43`, and `44` for headline runs; all data shuffling, weight initialization, and augmentation generators seeded.
2. **Confidence Intervals:** 95% confidence intervals generated via 1,000 paired bootstrap iterations over speakers/utterances. Competing models are evaluated against identical resample masks.
3. **Dual minDCF Reporting:** Every evaluation script must calculate and output minDCF at both:
   - $P_{\text{target}} = 0.01$ ($C_{\text{miss}} = 1.0, C_{\text{fa}} = 1.0$)
   - $P_{\text{target}} = 0.05$ ($C_{\text{miss}} = 1.0, C_{\text{fa}} = 1.0$)

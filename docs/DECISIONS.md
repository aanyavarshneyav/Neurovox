# DECISIONS.md — Architecture & Design Decision Records (ADR)

This document records the architectural, methodological, and experimental design choices for the SKA-HNN and IAAMSoftmax reproduction project.

---

## ADR-001: AdaptiveDrop Cleansing Schedule, Sequence Ordering & Safety Guards
- **Date:** 2026-10-28
- **Status:** Accepted
- **Context:** The draft PRD set $ESD=5$ and $ESC=7$, causing sample dropping/filtering to start before label correction. This contradicted the paper's core rationale ("we alleviate the negative impact of over-cleaning since we first correct suspicious labels before filtering them") and Algorithm 1. In addition, the paper omitted the mathematical formula for label correction and the $ESC$ threshold, while filtering without a minimum retention threshold risks batch starvation if all sample cosine similarities fall below $\tau=0.423$.
- **Decision:**
  1. Set warm-up period to 5 epochs: $ESD = 5$ `[Paper]`.
  2. Keep label correction active per user direction; set $ESC = 5$ so that correction operates prior to or concurrently with filtering `[Engineering Choice]`.
  3. Formula for label correction: if $CS(x, y) < \tau$ and $\max_{c \neq y} CS(x, c) \ge \tau_{corr}$ (default $\tau_{corr} = 0.5$), assign $\hat{y} = \arg\max_c CS(x, c)$. Tag this rule as `[External-UNVERIFIED] / [Engineering Choice]`.
  4. Implement an empty-batch safety guard: if filtering retains fewer than $\max(1, \lfloor 0.5 N \rfloor)$ samples, retain the top-$k$ highest cosine similarity samples to prevent gradient crash.
- **Consequences:** Removes the sequence inversion and gives a defined behavior when a batch is fully filtered. The paper's own Algorithm 1 does not specify the ESC value or the correction rule, so ESC=5 and the correction rule remain unverified choices (OPEN_QUESTIONS Q1). **Amendment (2026-10-28, after reading the paper text):** (a) the quote above is verbatim from Sec. II; (b) the text uses CS ≥ τ and Algorithm 1 uses CS > τ; we use ≥; (c) the paper does not describe a minimum-keep guard, so `min_keep_ratio` (default 0.5) is our addition and, because it forces retention under heavy noise, it is an ablation switch that can be disabled; its trigger count is logged.

---

## ADR-002: Dual Baseline Capacity Evaluation for Tier 0
- **Date:** 2026-10-28
- **Status:** Accepted
- **Context:** The paper's Table II compared ablations with large differences in parameter counts (e.g., S1 at 30.91M vs. S6 at 6.34M `[Paper, Table II]`) without controlling for capacity. A scaled SKA-HNN compared against an arbitrary smaller baseline would repeat that confound.
- **Decision:** Evaluate two ECAPA-TDNN baselines side-by-side in Tier 0:
  1. *Parameter-matched ECAPA-TDNN:* width chosen so its measured count is close to the measured scaled SKA-HNN count (width and count TBD: measure with torchinfo) `[Engineering Choice]`.
  2. *Standard ECAPA-TDNN:* Standard SpeechBrain configuration ($C=1024$, literature reference ~14.7M; count TBD: measure with torchinfo) `[External-UNVERIFIED]`.
  Both parameter counts and performance metrics will be measured and reported side-by-side.
- **Consequences:** Distinguishes architectural inductive bias from raw parameter capacity, ensuring scientific defensibility.

---

## ADR-003: Leakage-Free SSL Protocol via Synthetic Noise & WavLM K-Means
- **Date:** 2026-10-28
- **Status:** Accepted
- **Context:** Generating pseudo-labels via K-Means on embeddings extracted from a supervised Tier 0 model constitutes supervised label leakage. Clustering raw MFCC statistics clusters by recording environment and chapter rather than speaker identity.
- **Decision:** Adopt a two-part evaluation strategy for Tier 3:
  1. *Part A (Primary Scientific Focus):* Inject controlled symmetric label noise ($10\%, 20\%, 40\%$) into clean labels. Because ground truth noise rate is known exactly, this cleanly isolates the AdaptiveDrop cleansing mechanism from clustering artifacts.
  2. *Part B (Realistic SSL Proxy):* Extract frozen self-supervised representations using `WavLM-Base+` (trained without speaker labels) and cluster into $K=251$ pseudo-speakers via K-Means. Log cluster purity and NMI against true speaker IDs.
- **Consequences:** Eliminates label leakage, provides an exact mathematical benchmark for noise robustness, and tests realistic clustering noise.

---

## ADR-004: Dual-Layer Embedding Extraction Evaluation (FC1 vs. FC2)
- **Date:** 2026-10-28
- **Status:** Accepted
- **Context:** Verified against the paper text: Sec. II-D / Appendix says embeddings are extracted "from the fully connected layer closest to the pooling layer", and the Fig. 3 caption repeats this, so FC1 is the stated embedding layer `[Paper]`. Table VIII also shows a second FC+BN before IAAMSoftmax. An earlier draft of this ADR called this an ambiguity; it is not.
- **Decision:** FC1 is the primary embedding `[Paper]`. FC2 is scored as an extra logged diagnostic only `[Engineering Choice]`, as you approved.
- **Consequences:** Faithful to the paper, with a cheap diagnostic for free.

---

## ADR-005: Strict Cross-Chapter Target Trials with Single-Chapter Speaker Exclusion
- **Date:** 2026-10-28
- **Status:** Accepted
- **Context:** In LibriSpeech, utterances recorded within the same chapter share microphone distance, room acoustics, and noise tone. Several speakers in `test-clean` appear in only one chapter. Falling back to same-chapter pairs allows models to cheat via room acoustic matching.
- **Decision:** Discard single-chapter speakers from genuine target trials during evaluation manifest generation. 100% of target trials must be strictly cross-chapter.
- **Consequences:** Guarantees biometric speaker verification across recording sessions and eliminates channel-matching shortcuts.

---

## ADR-006: Lightweight On-The-Fly Noise Augmentation Scheme
- **Date:** 2026-10-28
- **Status:** Accepted
- **Context:** The full MUSAN corpus is ~11 GB and creates disk and download overhead on local machines. In contrast, RIRS_NOISES is ~1.3 GB and provides genuine room impulse responses.
- **Decision:** Download the 1.3 GB RIRS_NOISES archive for reverberation. Replace the 11 GB MUSAN download with an on-the-fly noise synthesizer that mixes in-batch multi-speaker babble (SNR 10–20 dB) and Gaussian colored noise (white/pink, SNR 5–15 dB), combined with SpecAugment and speed perturbation.
- **Consequences:** Keeps disk usage under 10 GB while providing diverse acoustic perturbation.

---

## ADR-007: Apple Silicon Mac M3 Pro Hardware Profile & MPS Numerics
- **Date:** 2026-10-28
- **Status:** Accepted
- **Context:** The target compute platform is an Apple Mac M3 Pro chip. PyTorch MPS acceleration does not support `torch.cuda.amp.GradScaler`, and half-precision variance computation ($\mathbb{E}[x^2] - \mu^2$) suffers catastrophic cancellation.
- **Decision:**
  1. Build fully device-agnostic PyTorch code (`mps`, `cpu`, `cuda`).
  2. Compute all local statistics in `torch.float32` and clamp variance $\ge 10^{-5}$ before `sqrt`.
  3. Run margin loss and classification head in `torch.float32`.
  4. Size DataLoader workers for macOS (`num_workers=2`, `persistent_workers=True`, `pin_memory=False`).
- **Consequences:** Rock-solid numerical stability on Apple Silicon without NaN crashes.

---

## ADR-008: Elimination of Logarithmic EER Scaling Law
- **Date:** 2026-10-28
- **Status:** Accepted
- **Context:** The draft PRD cited an empirical regression $EER \approx 9.15 - 1.08 \ln N$ predicting negative EER at $N=5994$ and disagreeing with its own data points.
- **Decision:** Completely discard the regression formula. Present the performance gap between LibriSpeech and VoxCeleb qualitatively based on: (1) training speaker volume (251 vs 5,994), (2) acoustic domain difference (clean read audiobooks vs wild YouTube interviews), and (3) network width scaling ($C=32, C_1=256$ vs $C=128, C_1=512$).
- **Consequences:** Removes indefensible arithmetic and prepares a scientifically rigorous defense.

---

## ADR-009: Repository path convention
- **Date:** 2026-10-28
- **Status:** Accepted
- **Context:** PRD_review notes inconsistent paths (`scripts/prep_librispeech.py` vs `src/data/prep_librispeech.py`).
- **Decision:** Reusable code lives under `src/` (package `src.*`). Thin command-line entry points live under `scripts/` and only parse args/config and call `src`. Configs live under `configs/`. `[Engineering Choice]`
- **Consequences:** Docs and workflows refer to `scripts/<name>.py` for CLIs and `src/<pkg>/<name>.py` for library code.

---

## ADR-010: Provenance audit using the extracted paper text
- **Date:** 2026-10-28
- **Status:** Accepted
- **Context:** The first Phase 0 docs tagged items `[Paper]` without having read the PDF. The text was then extracted with pypdf (installed in the project `.venv`; scratch text kept outside the repo) and every `[Paper]` tag was re-checked.
- **Decision:** `[Paper]` is used only for items found in the paper text, tables, algorithm or appendix. Items from the PRD or companion work are `[External-UNVERIFIED]`. Items absent from the paper and chosen by us are `[Engineering Choice]`.
- **Consequences:** `d_attn`, SKA branch details, SE ratio, LS window, kernels/strides/dilations, SpecAugment mask widths, max LR, Adam betas and the correction rule are tagged as not in the paper.

---

## ADR-011: Dedicated Apple Silicon Mac M3 Pro Training Hardware & Environment
- **Date:** 2026-10-29
- **Status:** Accepted
- **Context:** Initial planning documents contemplated Google Colab (T4/A100) as secondary compute. The experimental execution will run entirely on local Apple Silicon hardware.
- **Decision:** Target local Apple Mac M3 Pro exclusively (`device = mps` with CPU fallback). No Colab compute will be used. Verified hardware environment:
  - Machine: MacBook Pro 14-inch (Model Mac15,6 MRX33HN/A)
  - CPU / GPU: Apple M3 Pro, 11 cores (5 Performance, 6 Efficiency)
  - Unified Memory: 18.00 GB (19,327,352,832 bytes via `sysctl hw.memsize`)
  - OS: macOS 26.5 (Build 25F71)
  - Available Storage: 28.02 GB free disk space
  - Software Stack: Python 3.12.5, PyTorch 2.15.0, TorchAudio 2.11.1
- **Consequences:** Eliminates external cloud dependencies, session disconnects, and cloud storage transfer overheads. Experiment budgets and batch sizes must be calibrated strictly to M3 Pro MPS throughput and memory constraints.

---

## ADR-012: Course Policy Alignment on AI Code Generation
- **Date:** 2026-10-29
- **Status:** Accepted
- **Context:** Earlier PRD drafts included strict academic boilerplate ("MUST WRITE YOURSELF") to avoid academic integrity issues. Clarification from the user confirmed that the course has no policy restricting AI usage, and AI-written code is fully permitted across all architectural and pipeline components.
- **Decision:** Remove and relax all "must write yourself" language from project documentation. Antigravity and pair-programming agents may synthesize model components, losses, data loaders, and evaluation routines directly, provided code correctness, typing, and test standards are rigorously upheld.
- **Consequences:** Clarifies engineering permissions and removes obsolete constraints on code generation.

---

## ADR-013: Mandatory MPS CPU Fallback Flag (`PYTORCH_ENABLE_MPS_FALLBACK=1`)
- **Date:** 2026-10-29
- **Status:** Accepted
- **Context:** PyTorch's Metal Performance Shaders (MPS) backend does not implement all PyTorch operators. By default, hitting an unsupported kernel throws a terminating `NotImplementedError`.
- **Decision:** Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the environment for all benchmark, training, and evaluation scripts.
- **Consequences:** Unsupported operations automatically execute on CPU while logging a warning, guaranteeing pipeline continuity and preventing catastrophic crash on rare operators.

---

## ADR-014: Dual Compute Infrastructure & Configurable Storage Paths (Colab T4 Support)
- **Date:** 2026-10-29
- **Status:** Accepted
- **Context:** In addition to the local Apple Silicon Mac M3 Pro, training can also leverage Google Colab (Tesla T4 GPU with up to 200 compute units available). Running workloads interchangeably across local and cloud environments requires robust, device-agnostic execution, automatic accelerator selection (`cuda > mps > cpu`), precision handling (fp16 with GradScaler on CUDA, fp32 on MPS), and flexible storage path resolution without hardcoding local directories (`/Users`) or cloud directories (`/content`).
- **Decision:**
  1. Support dual execution targets: Apple Mac M3 Pro (`device = mps`) and Google Colab (`device = cuda`). System comparisons must be conducted on identical hardware and precision.
  2. Implement centralized path resolution via `src.utils.paths` respecting environment variables `DATA_ROOT`, `CACHE_ROOT`, and `CKPT_ROOT`. Default to repo-relative paths on macOS and `/content/*` on Colab.
  3. Provide `notebooks/colab_runner.ipynb` for idempotent Google Drive mounting, single-file archive transfer to fast local SSD, environment checks, and continuous checkpoint synchronization back to Drive.
  4. Allocate experimental tiers across platforms based on empirical throughput benchmarks and compute unit efficiency.
- **Consequences:** Full portability and flexibility between local and cloud training runs, with zero hardcoded filesystem paths and rigorous tracking of hardware and precision in results logs.

---

## ADR-015: Training Epoch Definition, Batch Sampling Protocol, and Schedule Progression
- **Date:** 2026-10-31
- **Status:** Accepted
- **Context:** In speaker recognition, training batch formulation directly affects margin loss optimization and curriculum schedules. A draft PRD stated that an epoch consists of "visiting all 251 speakers once". With batch size $B=64$, visiting 251 speakers would yield $\lceil 251 / 64 \rceil \approx 4\text{ batches}$ per epoch. A 4-batch epoch would process only $\approx 256$ utterances out of the 28,539 utterances in LibriSpeech `train-clean-100` ($<1\%$ dataset coverage), rendering epoch-based scheduling (learning rate decay, validation checks, and AdaptiveDrop warmup $ESD=5$ / $ESC=5$) completely broken and ineffective.
- **Decision:**
  1. Define 1 "training epoch" as exactly one complete pass through the training utterance volume: $\lceil 28,539 / 64 \rceil = \mathbf{446\text{ batches}}$ of size $B=64$ (28,539 utterances per epoch) `[Engineering Choice]`.
  2. Batch formulation: In each batch of 64, sample 64 distinct speakers randomly without replacement from the 251 training speakers, taking one random 180-frame utterance crop per speaker `[Engineering Choice]`. Across 446 batches, utterances are visited uniformly while ensuring all 64 items within any batch belong to distinct speaker classes (preventing duplicate-speaker collisions within mini-batch margin loss).
  3. AdaptiveDrop scheduling: The warmup threshold ($ESD=5$) and label correction threshold ($ESC=5$) increment in terms of these 446-batch epochs (i.e. $5 \times 446 = 2,230\text{ gradient steps}$) `[Engineering Choice]`.
  4. Explicitly reject the draft PRD's ~4-batch "visit all 251 speakers" wording as invalid for dataset coverage.
- **Consequences:** Guarantees proper stochastic gradient descent, uniform speaker coverage, full dataset exposure each epoch, and predictable AdaptiveDrop curriculum progression.

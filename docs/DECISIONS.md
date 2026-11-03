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

---

## ADR-016: LibriSpeech Verification Trial Volume & Uniform Sampling Protocol
- **Date:** 2026-10-31
- **Status:** Accepted
- **Context:** Standard LibriSpeech test partitions (`test-clean` with 40 speakers, `test-other` with 33 speakers, `dev-clean` with 40 speakers) contain both multi-chapter and single-chapter speakers. Demanding an arbitrary hard-coded 30,000 target trials would require ~750–1,000 pairs per speaker, causing heavy utterance reuse (>15x–30x per utterance) and exceeding the total available cross-chapter pairs for speakers with fewer utterances. Furthermore, 9 out of 40 speakers in `dev-clean` (22.5%) appear in only a single chapter, making genuine cross-chapter pairing impossible for them.
- **Decision:**
  1. Single-chapter speakers are strictly excluded from genuine target trials (preserving 100% cross-chapter integrity per ADR-005). They are retained for gender-matched impostor trials `[Engineering Choice]`.
  2. The target trial count is dynamically calibrated to what the data naturally supports without heavy reuse: sampling targets uniformly across qualifying speakers (capped at ~100 pairs per speaker, yielding ~3,100 targets and 3,100 gender-matched impostors = 6,200 trials for `dev-clean` and similar for `test-clean`), keeping mean utterance reuse under ~4–5x `[Engineering Choice]`.
  3. Every trial generation run writes a comprehensive audit report (`trials_report.json`) documenting excluded speakers, fractions, per-speaker target/impostor counts, reuse histograms, and SHA-256 checksums `[Engineering Choice]`.
- **Consequences:** Eliminates room-tone acoustic cheating, avoids artificial sample duplication, and provides complete cryptographic transparency and reproducibility for verification evaluation.

---

## ADR-017: Verification Trial Protocol v2 — Gender Matching, Cross-Chapter Utterance-Weighted Impostors, Exclusion Symmetry, and Cryptographic Supersession
- **Date:** 2026-11-03
- **Status:** Accepted (supersedes v1 trial lists and ADR-016 single-chapter impostor retention)
- **Context:**
  1. An independent methodological review (`docs/reviews/1B_review.md`) identified that ADR-016 had retained single-chapter speakers in the impostor pool while excluding them from targets. Because single-chapter speakers in LibriSpeech exhibit significantly longer mean utterance durations (e.g., 9.19 s in `test-clean` vs 7.01 s for multi-chapter speakers), this created an acoustic duration shortcut (11.54% target-impostor duration difference).
  2. Impostor gender counts were not matched to target gender counts (H4), resulting in gender distribution skew.
  3. Chapter-uniform impostor sampling distorted the utterance marginal distribution relative to cross-chapter target sampling (H5).
  4. ADR-016 claimed utterance reuse was capped at 4–5x, but that held only for the *mean*; due to asymmetric chapter sizes within multi-chapter speakers (e.g., 2 utterances in chapter A vs 35 in chapter B), capping maximum per-utterance reuse at $\le 10$ while enforcing 100 cross-chapter targets per speaker is mathematically impossible on LibriSpeech.
- **Decision:**
  1. **Exclusion Symmetry:** Single-chapter speakers are strictly excluded from BOTH target trials and impostor trials. This restores symmetric speaker representation and eliminates the single-chapter duration shortcut `[Engineering Choice]`.
  2. **Exact Gender Matching (H4):** Impostor gender counts ($N_{\text{imp}, M}, N_{\text{imp}, F}$) are matched exactly to target gender counts ($N_{\text{tgt}, M}, N_{\text{tgt}, F}$) per partition `[Engineering Choice]`.
  3. **Utterance-Weighted Impostor Sampling (H5):** Impostor utterances are sampled with cross-chapter pair weights identical to target marginals, guaranteeing that target-vs-impostor mean duration differs by $< 3\%$ (measured: $0.33\%$ on `test-clean`, $2.48\%$ on `test-other`, $1.93\%$ on `dev-clean`) `[Engineering Choice]`.
  4. **Deterministic Sorting & Local RNG:** Manifest rows are sorted by `utt_id` prior to processing, and per-call local RNG instances are used so output is bitwise deterministic regardless of execution environment `[Engineering Choice]`.
  5. **Companion Metadata CSV:** Every trial list is accompanied by a `_meta.csv` file recording `spk_a`, `spk_b`, `gender`, `dur_a`, `dur_b`, and utterance IDs for speaker bootstrap evaluation `[Engineering Choice]`.
  6. **Utterance Reuse Transparency:** While mean (4.5x–5.9x) and median (4.0x–5.0x) reuse remain modest, the full empirical distribution and percentiles are logged in `trials_report_*.json`, superseding the informal ADR-016 claim `[Engineering Choice]`.
  7. **Cryptographic Supersession:** The v1 trial lists are marked SUPERSEDED:
     - `trials_test-clean.txt`: v1 (`371a42e7dee37d6cfa5a61b7b264e3cb53c68680c5499c038627e2eede15f228`) $\to$ superseded by v2 (`5b47db5e847990a75b35edfc5d906944dd069a59144df7c6b5f5126388ee6aa1`).
     - `trials_test-other.txt`: v1 (`fa3b756a9c3b9b567126e37abd12021192922ece779385663d450543262517a6`) $\to$ superseded by v2 (`92e651bccc3af6253b6c97437d3197812cf384490cbe16cd812253779651a9a6`).
     - `trials_dev-clean.txt`: v1 (`e155f12922d345f6d730cfcbd32d3082da55b4e26e105e062771ea814532002c`) $\to$ superseded by v2 (`2025424ac8116efa2ab5a008cb112d4598a7c2e8b28f30abb6be81680312d32d`).
- **Consequences:** Restores complete duration and gender symmetry, eliminates acoustic duration shortcuts, enables speaker-level bootstrap CIs via exported metadata, and documents the true empirical utterance reuse distribution.


---

## ADR-018: Feature Extraction, Augmentation Architecture, and Audio Caching Strategy
- **Date:** 2026-11-03
- **Status:** Accepted
- **Context:** Phase 1C implements the acoustic front-end (80-dim log-Mel filterbanks, CMN, SpecAugment), data loader pipeline with random 180-frame cropping (wrap padding for short clips), waveform augmentations (speed perturbation, colored noise, babble, RIR), and speaker-balanced mini-batch sampling.
- **Decision:**
  1. **On-the-Fly Audio Decoding (No Disk Memmap Cache by Default):**
     Measured data-loading latency on genuine LibriSpeech audio (`train-clean-100`) with $B=64$, 4 DataLoader workers, and active augmentations is **$47.36\text{ ms/batch}$** ($21.12\text{ batches/s}$). This is **2.46x faster** than the lower-bound model forward+backward step ($116.51\text{ ms}$) and **8.69x faster** than the full SKA model step ($411.52\text{ ms}$). (*Note: Earlier exploratory tests indicated a ~6.1x throughput figure for disk memmap caching, but this was not a like-for-like comparison as the memmap test evaluated $B=32$ on a 50-speaker subset held entirely within OS page cache RAM vs. full-dataset on-the-fly decoding.*) Because asynchronous background worker pre-fetching completely hides I/O latency, and building an ~11.6 GB int16 memmap cache on disk would leave only $< 0\text{ GB}$ free space (violating the $\ge 8\text{ GB}$ free disk buffer requirement on the Mac M3 Pro), on-the-fly decoding is retained as the default strategy. The disk memmap writer and reader (`scripts/build_audio_cache.py`, `src/data/cache.py`) are fully implemented and tested if needed `[Engineering Choice]`.
  2. **Speed Perturbation Optimization:**
     Continuous speed factors $\in [0.95, 1.05]$ when passed to `torchaudio.functional.resample` cause GCD drops (e.g., $\text{gcd}(16552, 16000) = 8$), triggering massive polyphase FIR filter synthesis on CPU (~666 ms per utterance). Speed perturbation is therefore performed via 1D linear interpolation (`torch.nn.functional.interpolate` with `mode="linear"`) on a pre-cropped $\approx 33,000$-sample audio buffer, executing in $< 0.25\text{ ms}$ with high acoustic fidelity `[Engineering Choice]`.
  3. **Strict Worker-Seeded Augmentations:**
     All waveform and spectrogram augmentations receive a local `np.random.Generator` derived from `worker_info.seed` (or base seed), the current epoch, and the sample index. No augmentation or random crop relies on Python's global `random`, global `np.random`, or global PyTorch RNG `[Engineering Choice]`.
  4. **Speaker-Balanced Batch Sampler:**
     Mini-batches of size $B=64$ sample $B$ distinct speakers without replacement. An unvisited speaker pool ensures all 251 training speakers are guaranteed to be visited every epoch before any speaker is re-visited `[Engineering Choice]`.
- **Consequences:** End-to-end non-blocking data loading ($47.36\text{ ms} \ll 411.52\text{ ms}$), zero disk exhaustion risk, completely reproducible worker-level augmentation streams, and guaranteed distinct speakers per batch.

---

## ADR-019: Google Colab Dual-Platform Infrastructure, Dynamic Environment Path Routing, Token Sanitization, and Cross-Machine Cryptographic Verification
- **Date:** 2026-11-03
- **Status:** Accepted
- **Context:**
  1. The project adopts a dual-platform development workflow: local interactive development and debugging on Apple Silicon Mac M3 Pro (`device = mps`, `torch.float32`), while running the extensive training and evaluation grid on Google Colab (Tesla T4 GPU, ~2 vCPU cores, fresh VM each session, ~200 compute units available).
  2. Running identical code across both environments requires dynamic directory resolution without machine-specific hardcoded paths (`/Users` or `/content`).
  3. Colab VMs are ephemeral: private code access must use secrets without persisting tokens in git configurations, and training artifacts (checkpoints, logs, result JSONs) must be mirrored to Google Drive every epoch.
  4. Cross-machine reproducibility requires mathematical verification that data preparation and trial list generation on a fresh cloud VM produce bitwise identical outputs to the committed reference lists.
- **Decision:**
  1. **Colab Requirements Decoupling (`requirements-colab.txt`):** PyTorch and TorchAudio are intentionally excluded from `requirements-colab.txt` because Google Colab provides pre-installed, hardware-optimized CUDA PyTorch builds. Overwriting them with standard pip wheels risks CUDA driver incompatibilities or broken GPU acceleration `[Engineering Choice]`.
  2. **Environment Variable Path Routing (`src/utils/paths.py`):** Path resolution is dynamically governed by `DATA_ROOT`, `CACHE_ROOT`, `CKPT_ROOT`, and `RESULTS_ROOT`. `resolve_path()` automatically inspects the root component of relative paths and routes `"data"`, `"cache"`, `"checkpoints"`, and `"results"` to their respective environment-configured roots (defaulting to project subdirectories on Mac and `/content/*` on Colab). Hardcoded `/Users` paths are strictly forbidden and enforced by unit tests `[Engineering Choice]`.
  3. **Credential Sanitization:** The runner notebook (`notebooks/colab_runner.ipynb`) retrieves `GITHUB_TOKEN` from Colab Secrets (`userdata.get`) for authenticated cloning/pulling, but immediately executes `git remote set-url origin https://github.com/...` to strip the secret token from the saved git configuration on disk `[Engineering Choice]`.
  4. **Cross-Machine Reproducibility Gate (`scripts/verify_trials.py`):** On every fresh VM session, trial lists regenerated by `trial_generator.py` are cryptographically verified against the committed ADR-017 SHA-256 hashes (`5b47db5e...` for `test-clean`, `92e651bc...` for `test-other`, `2025424a...` for `dev-clean`, and companion metadata CSVs). Any checksum divergence raises a fatal `RuntimeError` and terminates execution before training begins `[Engineering Choice]`.
  5. **Device-Agnostic Benchmarks & Compute Budget Tracking:** Hardware and dataloader benchmark scripts dynamically detect CUDA > MPS > CPU. On CUDA, both `fp32` and `fp16` with `torch.cuda.amp.GradScaler` are benchmarked for the full-SKA composite, and dataloader worker sweeps support `{0, 1, 2}` workers. Colab compute unit consumption is budgeted using $\text{runs} \times \text{epochs} \times (\text{sec\_per\_epoch} / 3600) \times \text{units\_per\_hour}$ `[Engineering Choice]`.
- **Consequences:** Seamless, zero-leakage cross-machine portability, guaranteed bitwise cross-platform trial list parity, safe Drive checkpoint recovery with `--resume`, and clear compute unit expenditure tracking.

---

## ADR-020: Phase 2A Baseline Pipeline, Full-Utterance Validation Mechanics, Margin Loss Implementation, and Training-Data EER Monitoring
- **Date:** 2026-11-03
- **Status:** Accepted
- **Context:**
  1. Phase 2A establishes the end-to-end training and evaluation pipeline on a fast, lightweight baseline model (`small_tdnn_se`, 0.411M parameters) prior to introducing the full SKA-HNN composite backbone.
  2. Training requires a stateful training loop with optimizer/scheduler/RNG persistence, per-step NaN/Inf interceptors, cyclic learning rate scheduling, and strict speaker-balanced batching ($B=64$ distinct speakers, 446 batches/epoch per ADR-016).
  3. Margin loss formulation: AAMSoftmax ($s=30, m=0.2$, margin applied exclusively to target class $j = y_i$, denominator summing over all classes $j$) requires high-precision execution (`fp32`) on MPS to avoid numerical overflow when computing margins $\cos(\theta + m)$, with a monotonic fallback when $\theta + m > \pi$.
  4. Evaluation protocol: Speaker verification evaluation must operate over FULL audio utterances (without temporal cropping) in `model.eval()` mode with instance-level Cepstral Mean Normalization (CMN), using default batch size 1 (or identical-length batching) to guarantee zero-padding batch invariance ($< 1\times 10^{-4}$ tolerance).
  5. Generalization vs. Memorization tracking: To continuously monitor the train-versus-dev gap and detect overfitting on 251 speakers without touching test sets, a fixed-seed training-data probe trial list (`trials_train-probe.txt`, 5,852 cross-chapter trials from `train-clean-100`) is evaluated alongside `dev-clean`.
- **Decision:**
  1. **Model Decoupling:** Implement `small_tdnn_se` (1D TDNN + Squeeze-and-Excitation + clamped stats-pooling + projection to 192-dim) exclusively as a pipeline validation vehicle. Parameter counts are strictly measured via `torchinfo` (410,928 parameters), not hand-calculated `[Engineering Choice]`.
  2. **Loss Numerics & Overflow Guard:** AAMSoftmax computes in `torch.float32`. When $\theta + m > \pi$, the cosine margin is replaced with monotonic linear fallback $\cos(\theta) - m \cdot \sin(\pi - m)$ to prevent gradient inversion. Unit tests verify exact mathematical parity ($< 1\times 10^{-5}$) against a 64-bit reference formula `[Engineering Choice]`.
  3. **Zero-Leakage Evaluation Pipeline:** `EmbeddingExtractor` extracts embeddings strictly for utterances participating in trial lists, caches embeddings in memory, and asserts $L_2$ unit-norm normalizations. Evaluator computes cosine similarity, threshold-free EER, normalized minDCF at $P_{\text{target}} \in \{0.05, 0.01\}$, and speaker-level clustered bootstrap confidence intervals (1,000 iterations). `test-clean` and `test-other` remain strictly untouched `[Engineering Choice]`.
  4. **RNG & Checkpoint Parity:** Trainer checkpoints save model, loss, optimizer, scheduler, and all 4 system RNG states (`torch`, `numpy`, `python`, `mps`/`cuda`). Resume parity tests confirm bitwise/tight match ($< 1\times 10^{-5}$ CPU, $< 1\times 10^{-4}$ MPS) between uninterrupted runs and resumed runs `[Engineering Choice]`.
  5. **Dual Evaluation Trajectory:** Every epoch logs `dev-clean` EER alongside `train-probe` EER (labeled explicitly as training-data EER), separating `train_sec` from `eval_sec` and exporting complete histories to JSON/CSV artifacts `[Engineering Choice]`.
- **Consequences:** Provides a rock-solid, mathematically verified pipeline that guarantees zero biometric leakage, full resume determinism, robust numeric safety guards, and immediate visibility into generalization gaps before scaling to Tier 1 SKA-HNN models.

### Amendment (2026-11-03): Batch Sampler `set_epoch` Bug Resolution & v2 Verification (ADR-020.1)
- **Problem Statement:** During Phase 2A peer review, diagnostic testing revealed that `Trainer.train_epoch` called `set_epoch` on `train_loader.sampler`. Because PyTorch's `DataLoader` assigns a default `SequentialSampler` to `.sampler` when `batch_sampler` is provided, `SpeakerBalancedBatchSampler.set_epoch` was never invoked. Furthermore, under `persistent_workers=True`, modifications to `dataset.set_epoch()` in the main process did not propagate to child worker processes. Consequently, historical Runs B and C replayed the exact same batch formation and identical crop/augmentation seeds across all epochs, limiting total training data exposure to only 63.23% unique utterances (18,045 / 28,539) across 10 epochs.
- **Decision:**
  1. **Direct `batch_sampler` Epoch Advance:** In `Trainer.train_epoch`, verify `train_loader.batch_sampler` has a `set_epoch` method (raising `TypeError` otherwise) and invoke `batch_sampler.set_epoch(epoch)`.
  2. **Tuple Item Passing & Worker RNG Isolation:** `SpeakerBalancedBatchSampler` yields `BatchItem(idx, epoch)`. `LibriSpeechDataset.__getitem__` unpacks `(idx, epoch)` and derives its local RNG strictly from `np.random.SeedSequence([self.base_seed, epoch, sample_idx])`, eliminating any dependence on worker PID, worker count, or persistent worker process memory.
  3. **Sampler Diagnostics & Checkpoint Validation:** In `history.csv`, log `sampler_epoch`, `first_batch_index_hash`, `unique_utts_this_epoch`, and `cum_unique_utts` every epoch.
  4. **Status of Historical Runs:** Mark historical `sanity_2a_50spk` and `sanity_2a_full` as SUPERSEDED. Establish `sanity_2a_overfit_v2`, `sanity_2a_50spk_v2`, and `sanity_2a_full_v2` as authoritative benchmarks (achieving 99.99% cumulative utterance coverage across 10 epochs).

### Amendment (2026-11-03): Sampler Draw-ID 3-Tuple Contract, Strict Index Guard, and v3 Baseline Runs (ADR-020.2)
- **Problem Statement:** During Phase 2A review, edge cases in seeding and data contracts were identified:
  1. *Identical draw collision:* If an utterance `idx` was drawn multiple times in the same epoch (e.g. in small datasets or long epochs), using `(base_seed, epoch, idx)` yielded identical random crops and augmentations for both draws.
  2. *Silent indexing fallback:* Calling `dataset[idx]` with a plain int in train mode would silently use `self.epoch` rather than failing fast.
  3. *Unpurged overwrite artifacts:* `--overwrite` did not purge obsolete checkpoints from prior runs.
  4. *CUDA RNG state restoration:* `torch.cuda.set_rng_state` requires a CPU-mapped ByteTensor; direct restore from GPU state could fail.
  5. *Metadata transparency:* Configured augmentation probabilities were not verified against dataset internals, and cyclic scheduler parameters were incompletely logged.
- **Decision:**
  1. **3-Tuple Sampler Contract & SeedSequence Isolation:** `SpeakerBalancedBatchSampler` yields `BatchItem(idx, epoch, draw_id)` where `draw_id` is an integer monotonically incremented for each draw within the epoch. `LibriSpeechDataset.__getitem__` unpacks `(idx, epoch, draw_id)` and generates worker RNG from `np.random.SeedSequence([base_seed, epoch, idx, draw_id])`. If an utterance is drawn twice in an epoch, distinct random transformations are guaranteed.
  2. **Strict Indexing Type Guard:** In train mode, passing a plain `int` to `dataset.__getitem__` strictly raises `TypeError`. Only evaluation mode permits integer indexing.
  3. **Strict Checkpoint/Results Purge on `--overwrite`:** `--overwrite` removes the run's checkpoint and results directories with `shutil.rmtree` before training begins.
  4. **CUDA RNG Host Mapping:** `Trainer.load_checkpoint` calls `.cpu()` on the serialized CUDA RNG state before restoring via `torch.cuda.set_rng_state`.
  5. **Wired Config Probabilities & Complete Metadata:** Real augmentation probabilities (`speed`, `noise`, `babble`, `rir`, `specaug`), `grad_clip`, `lr_schedule`, `dataset_size`, and `num_speakers` are wired and logged in `run_meta.json`.
  6. **Status of v2 Runs:** Mark v2 runs as `PROVISIONAL (superseded by v3)`. Establish `sanity_2a_overfit_v3`, `sanity_2a_50spk_v3`, and `sanity_2a_full_v3` as authoritative baselines.

## ADR-021: Google Colab Preflight Protocol, Pinned Commit Execution, and Controlled Training Pilot Protocol (Phase 3A)
- **Date:** 2026-11-04
- **Status:** Accepted
- **Context:**
  1. Moving from local development and sanity validation to full LibriSpeech experiments on cloud GPU (Google Colab) requires rigorous preflight verification to prevent non-reproducible runs, silent configuration drift, or wasted compute units.
  2. To guarantee bitwise cross-machine reproducibility, the Colab notebook must execute code strictly pinned to commit `720a010` (Phase 2 completion), rather than an unverified or outdated local branch.
  3. Colab VMs are ephemeral. Storage paths must strictly route data loading and feature caches to fast local VM NVMe disks (`/content/data`, `/content/cache`), while persisting checkpoints and results to Google Drive (`/content/drive/MyDrive/Neurovox_Sync`).
  4. Data and trial integrity must be established before committing GPU compute: partition audio files, speaker counts (251, 40, 40, 33), utterance counts (28,539, 2,703, 2,620, 2,939), 100% speaker disjointness, and 100% cross-chapter target guarantee must be verified. Existing valid trial manifests must NOT be regenerated.
  5. The training pipeline must be validated under a controlled 1-epoch pilot for both S0 (IAAMSoftmax with AdaptiveDrop) and S1 (AAMSoftmax) under full model capacity ($C=32, C_1=256$, 4.168M params), full batch size ($B=64$), and strict fp32 precision.
  6. AdaptiveDrop filtering and safeguard behavior across training epochs must be captured in history artifacts (`history.json`, `history.csv`) and verified.
- **Decision:**
  1. **Pinned Repository Execution:** `notebooks/colab_runner.ipynb` strictly checks out commit `720a010c3f356213f25b3fdb5b1b83139b50581d` and logs the commit hash and author details. Git commits are prohibited from Colab `[Engineering Choice]`.
  2. **Comprehensive Preflight Audit:** The notebook reports GPU model, available VRAM, CUDA/PyTorch versions, system RAM, available disk space, installed dependencies, and commit SHA before launching any training workload `[Engineering Choice]`.
  3. **Data Protocol & Trial Integrity Audit:** Integrated into `scripts/verify_trials.py --check-data`, verifying partition counts, 100% speaker disjointness across train, dev, test-clean, and test-other, and 100% cross-chapter target trials, alongside bitwise SHA-256 matching for all 8 trial files and 5 manifests `[Engineering Choice]`.
  4. **Phase 3A Controlled Pilot Configurations:** Configured `configs/pilot_s0_full.yaml` (`pilot_s0_epoch1`) and `configs/pilot_s1_full.yaml` (`pilot_s1_epoch1`). Each executes 1 complete epoch (446 batches of $B=64$ distinct speakers) on `train-clean-100`, followed by dev-clean full-utterance evaluation (6,200 trials) `[Engineering Choice]`.
  5. **Loss Diagnostics Propagation:** Updated `Trainer.fit()` to forward `loss_*` fields from `train_epoch()` into epoch records, ensuring `history.json` and `history.csv` capture `loss_num_corrected`, `loss_num_filtered`, `loss_num_retained`, `loss_keep_ratio`, and `loss_guard_activations` `[Engineering Choice]`.
  6. **Empirical Compute Budget Formulation:** Compute budgets for headline experiments (S0 vs S1 across seeds 42, 123, 777) are calculated directly from empirical epoch times measured in the pilot, rather than historical MacBook estimates `[Engineering Choice]`.
- **Consequences:** Eliminates silent reproducibility failure modes, establishes an automated audit trail before multi-epoch training, guarantees safe checkpoint persistence across Colab runtime drops, and provides accurate GPU compute unit forecasting.





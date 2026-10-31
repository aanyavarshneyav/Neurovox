# AGENTS.md — Standing Operational Rules

## 1. Project Identity & Scope (5-Line Summary)
- **Goal:** Reproduce the SKA-HNN backbone and IAAMSoftmax/AdaptiveDrop loss from Alam et al. (IJCNN 2025) on LibriSpeech.
- **Hardware Profile:** Dual platform — MacBook Pro (Apple M3 Pro, 11 cores, 18 GB RAM, macOS 26.5, `device=mps` with CPU fallback) and Google Colab (Tesla T4 GPU, CUDA). Run locations decided per experiment type based on measured benchmarks.
- **Policy & Code Integrity:** The course has no AI-use policy; AI-written code is permitted. Any historical "must write yourself" language is relaxed (see ADR-012).
- **Success Criteria:** Pipeline executes end-to-end without error, reproduces relative ablation rankings on clean and noisy data, and preserves strict scientific integrity.
- **Methodology & Baseline:** Scaled channel dimensions ($C=32, C_1=256$) to prevent overfitting on 251 speakers, with baseline capacities matched or reported side-by-side.
- **Tier Structure:** Tier 0 (Baselines: ECAPA-TDNN) → Tier 1 (Core SKA-HNN S0 & S1) → Tier 2 (Ablations S2–S6) → Tier 3 (Synthetic Noise & SSL WavLM K-Means) → Tier 4 (Extensions).

---

## 2. Repository Map & Documentation Index
- `AGENTS.md`: This file — concise standing rules and non-negotiable constraints.
- [docs/PROJECT_SPEC.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/PROJECT_SPEC.md): Corrected architectural and algorithmic specification with provenance tags.
- [docs/ARCHITECTURE.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/ARCHITECTURE.md): Module-by-module input/output shapes, tensor contracts, and ablation switches S0–S6.
- [docs/DATA_PROTOCOL.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/DATA_PROTOCOL.md): Splits, manifests, trial generation, leakage checks, and Mac M3 Pro caching plan.
- [docs/EXPERIMENT_PLAN.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/EXPERIMENT_PLAN.md): Five-tier roadmap, technical gates, hypothesis testing, and compute budgets.
- [docs/DECISIONS.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/DECISIONS.md): Architectural Decision Records (ADRs) logging rationale for choices made.
- [docs/OPEN_QUESTIONS.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/OPEN_QUESTIONS.md): Unverified claims, ambiguous paper details, and current assumptions.
- [docs/RESULTS_LOG.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/RESULTS_LOG.md): Append-only experimental ledger recording verified metrics per run.
- `.agents/rules/`: Specialized behavioral constraints (`00-integrity.md`, `10-python.md`, `20-experiments.md`).
- `.agents/workflows/`: Step-by-step procedures (`param-audit`, `new-experiment`, `pre-commit-check`, `session-handoff`).

---

## 3. Scientific Integrity Rules (Strictly Enforced)
1. **Measured Numbers Only:** NEVER state a number as an experimental result unless it was measured and logged in [docs/RESULTS_LOG.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/RESULTS_LOG.md) with an active `run_id` and config hash. Any anticipated or literature-derived number must be explicitly tagged `[HYPOTHESIS]`.
2. **Torchinfo Parameter Counts:** Parameter counts must NEVER be hand-calculated. All parameter figures must originate from `torchinfo.summary()` execution via `.agents/workflows/param-audit.md`. Until measured, mark counts as `"TBD: measure with torchinfo"`.
3. **Mandatory Provenance Tags:** Every specification claim, hyperparameter, and formula must carry an unambiguous provenance tag:
   - `[Paper]`: Verbatim or mathematically direct from Alam et al. (IJCNN 2025).
   - `[External-UNVERIFIED]`: Derived from external papers, theses, or unverified repos; must be tracked in [docs/OPEN_QUESTIONS.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/OPEN_QUESTIONS.md).
   - `[Engineering Choice]`: Deliberate design or scaling decisions made for this project, documented in [docs/DECISIONS.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/DECISIONS.md).
4. **Technical-Only Gates:** Go/no-go gates must be purely technical (zero NaNs, exact tensor shapes, passing unit tests, non-zero gradients). A gate must NEVER demand "result must improve" or "EER < X%".
5. **Report Negative Results:** Negative, null, or non-monotonic results must be reported immediately and recorded as-is without massaging or retrying until a hypothesis is confirmed.
6. **When Unsure, Ask:** DO NOT invent paper details, equations, or hyperparameter values. If an detail is missing from the paper, verify in `docs/source/Additive_Angular_Margin_Loss.pdf`, record the ambiguity in [docs/OPEN_QUESTIONS.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/OPEN_QUESTIONS.md), and ask the user.

---

## 4. Evaluation Rigor Rules
1. **Multi-Seed Headline Comparisons:** Any headline comparison (e.g., S0 vs S1, noise sweeps, baseline vs proposed) must evaluate $\ge 3$ random seeds (mean $\pm$ std).
2. **Clustered / Paired Bootstrap:** Compute 95% confidence intervals using paired bootstrap resampling over speakers or utterances (not individual trial lines). Competing models must be evaluated against identical bootstrap resample indices.
3. **Dual minDCF Priors:** Always report minimum Detection Cost Function at BOTH $P_{\text{target}} = 0.01$ AND $P_{\text{target}} = 0.05$ (with $C_{\text{miss}} = 1.0, C_{\text{fa}} = 1.0$).
4. **Parameter-Matched Baselines:** Baselines (Tier 0 ECAPA-TDNN) must be evaluated both parameter-matched to the scaled SKA-HNN and at standard width ($C=1024$), reporting measured `torchinfo` counts for both, to eliminate capacity confounding. No parameter figure is assumed before measurement.
5. **Hardware & Precision Consistency:** Comparisons between competing systems must use identical hardware and numerical precision. Explicitly log device name, compute platform, and precision per run in [docs/RESULTS_LOG.md](docs/RESULTS_LOG.md).

---

## 5. Numerics & Device Rules
1. **fp32 Local Statistics:** Sliding-window local statistics ($\mu_t, \sigma_t$) must always be computed in `torch.float32`. Apply `torch.clamp(var, min=1e-5)` before `torch.sqrt` to prevent catastrophic cancellation.
2. **Device-Agnostic Code:** All code must dynamically select `mps`, `cuda`, or `cpu` via `device = torch.device(...)`.
3. **MPS Safety Guardrails:** On Apple Silicon Mac M3 Pro (`mps`), do NOT use `torch.cuda.amp.GradScaler`. Loss computation and prototype angular margins run in `torch.float32`. Autocast is enabled only where verified stable.
4. **MPS Operator Fallback:** Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in run scripts. This enables PyTorch to execute operations unsupported by MPS on the CPU rather than crashing with `NotImplementedError`.
5. **Measured Machine Environment:** MacBook Pro 14-inch (Apple M3 Pro, 11 cores [5P+6E], Model Mac15,6 MRX33HN/A); OS: macOS 26.5 (Build 25F71); Memory: 18.00 GB (19,327,352,832 bytes via `sysctl hw.memsize`); Disk: 28.02 GB free; Python 3.12.5; PyTorch 2.15.0; TorchAudio 2.11.1.

---

## 6. Data Engineering & Integrity Rules
1. **Speaker-Disjoint Splits:** Speaker IDs in `train`, `dev`, `test-clean`, and `test-other` must be mutually disjoint. Unit tests must assert zero intersection across partition speaker sets.
2. **Strict Cross-Chapter Target Trials:** Target trials in evaluation manifests must pair utterances from different book chapters. Single-chapter test speakers must be strictly discarded from genuine target trials so that 100% of target trials are cross-chapter.
3. **Zero Label Leakage in SSL:** In Tier 3 self-supervised clustering, pseudo-labels must NEVER be generated using models trained with true speaker labels. Cluster pseudo-labels using frozen self-supervised foundation embeddings (WavLM-Base+) or unsupervised acoustic features. Log cluster purity and NMI against ground truth.

---

## 7. Workflow & Operational Rules
1. **Plan Multi-File Edits:** Write an implementation plan and identify impacted contracts before editing multi-file changes.
2. **Small Atomic Commits with Permission:** Propose commits in logical units with descriptive git commit messages, but NEVER execute any git commit without asking the user for explicit permission first.
3. **Test Before Done:** Run `pytest` and verify clean execution before declaring any development task complete.
4. **Log Decisions:** Whenever a design, scaling, or algorithmic choice is made, create an entry in [docs/DECISIONS.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/DECISIONS.md).
5. **External Resource Safety:** Always ask the user for confirmation before downloading files $> 1\text{ GB}$ or deleting existing files/directories.

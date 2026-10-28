# OPEN_QUESTIONS.md — Unverified Items & Architectural Ambiguities

This document tracks all unverified claims, missing hyperparameter values, and mathematical ambiguities identified in the reference paper (*Alam et al., IJCNN 2025*) and companion sources.

---

## 1. Unverified Paper Details & Ambiguities

| # | Item Name | Status | Ambiguity / Missing Detail in IJCNN 2025 Paper | Working Assumption `[Engineering Choice]` | Blocking? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1** | **Label Correction Equation & $ESC$ Threshold** | `[External-UNVERIFIED]` | Algorithm 1 includes `if epoch >= ESC: Correct pseudo-labels`, but the text and algorithm omit the mathematical formula for label correction and the value of $ESC$. A companion paper cites $\tau_{corr}=0.5$ and $ESC=7$. | Adopt $ESC=5, ESD=5$ (or $ESC=3, ESD=5$). For samples with $CS(x, y) < \tau$ and $\max_{c \neq y} CS(x, c) \ge \tau_{corr}$ (default $0.5$), reassign label to $\arg\max_c CS(x, c)$. | No (Defined in ADR-001) |
| **Q2** | **S1 → S5 Parameter Delta in Table II** | `[Paper Ambiguity]` | Table II reports S1 at 30.91M and S5 (vanilla 2D-CNN) at 17.37M `[Paper]` (delta 13.54M `[Derived]`). PRD_review argues the SKA-to-vanilla swap in the ResBlocks cannot account for it `[External-UNVERIFIED: reviewer estimate, not measured]`. Did S5 also change widths? | Treat as an unexplained published discrepancy. Measure S1 and S5 with `torchinfo` and log counts. | No (resolved via `param-audit`) |
| **Q3** | **MLASP Attention Dimension Reduction ($3d_i \to d_i$)** | `[Paper Ambiguity]` | Section II-C states weights are learned over $X_i^c \in \mathbb{R}^{3d_i \times T}$ and "reduced to match the dimensionality of $X_i$" before multiplication. Equation (2) presents scalar attention $\alpha_t$. | Use a two-layer bottleneck Conv1d: $\text{Conv1d}(3d_i, d_{attn}, 1) \to \text{Tanh} \to \text{BN} \to \text{Conv1d}(d_{attn}, d_i, 1) \to \text{Softmax}$ over time, with $d_{attn}=128$ (scaled) / $256$ (full). | No (Defined in ADR-001) |
| **Q4** | **minDCF Target Prior ($P_{target}$)** | `[Paper Ambiguity]` | The paper reports minDCF (e.g., 0.0430 at 0.65% EER `[Paper, Table II]`) without specifying $P_{target}$, $C_{miss}$, or $C_{fa}$. PRD_review suggests it looks closer to $P=0.05$ than 0.01 `[External-UNVERIFIED]`. | Report minDCF at BOTH $P_{target}=0.01$ and $0.05$ ($C_{miss}=C_{fa}=1$). | No (ADR-001 / PROJECT_SPEC §4) |
| **Q5** | **Suspect External GitHub URL** | `[External-UNVERIFIED]` | Draft PRD cited `readourlips/SKA-TDNN`, which is an unverified external repository. | Do not use or depend on external unverified forks; build modular PyTorch implementations from first principles and paper equations. | No |
| **Q6** | **TDNN Dilation & LSTM Directionality** | `[Paper Ambiguity]` | Section II-B specifies 1 TDNN + 1 LSTM in TDNN-LSTM and 3 TDNNs in Fully-TDNN, but omits kernel sizes, dilations, and LSTM directionality. | Use standard x-vector conventions: TDNN-LSTM uses Conv1d($k=3, d=1$) + unidirectional LSTM($C_1, C_1$); Fully-TDNN uses Conv1d with dilations $d=[1, 2, 3]$. | No |
| **Q7** | **Local Statistics Window Length** | `[Paper Ambiguity]` | Paper: statistics computed by "resampling input sequences and adjusting pooling window shift rates"; no window length. The 21-frame default comes from the PRD citing Kang et al. (ASRU 2021) `[External-UNVERIFIED]`. | W=21, stride 1, replicate padding `[Engineering Choice]` (verify against the cited work if available). | No |
| **Q8** | **Embedding Extraction Layer** | `[Resolved]` | Paper Sec. II-D and Fig. 3 caption: embeddings from the FC layer closest to pooling (FC1). | FC1 primary `[Paper]`; FC2 scored as a diagnostic only (ADR-004). | No |
| **Q9** | **WavLM-Base+ overlap with LibriSpeech** | `[External-UNVERIFIED]` | WavLM-Base+ pretraining data may include Libri-Light/LibriVox audio overlapping LibriSpeech. It is label-free but could favor speaker clustering on LibriSpeech. | Record in results notes; interpret Tier 3 Part B with this caveat. | No |
| **Q10** | **K=251 uses the true speaker count** | `[Engineering Choice]` | The paper also fixes K to the true speaker count (5,994 `[Paper]`). | Report a K sensitivity check alongside K=251. | No |
| **Q11** | **VoxCeleb1-O access and size (Tier 4)** | `[External-UNVERIFIED]` | PRD claims ~1.2 GB; registration may be needed. Violates the >1 GB rule without confirmation. | Ask you before downloading. | No (Tier 4 only) |
| **Q12** | **`.agents/rules` frontmatter keys** | `[External-UNVERIFIED]` | `alwaysApply`/`globs` keys were chosen without checking the Antigravity schema. | Verify against the `agy-customizations` skill before relying on activation modes. | No |
| **Q13** | **Several SKA/SE/SpecAugment/LR details are absent from the paper** | `[External-UNVERIFIED]` | SKA branch count/kernels/ratio, SE ratio, SpecAugment widths, LR peak, Adam betas, TDNN kernels/dilations, LSTM directionality, d_attn. | Tagged in PROJECT_SPEC; verify against SKA-TDNN (Mun et al., SLT 2023) when available. | No |

---

## 2. Maintenance Protocol

When an open question is resolved through empirical measurement, user clarification, or literature confirmation:
1. Update its status in the table above to `[Resolved]`.
2. Document the decision and rationale in [docs/DECISIONS.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/DECISIONS.md).
3. Update relevant specifications in [docs/PROJECT_SPEC.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/PROJECT_SPEC.md) and [docs/ARCHITECTURE.md](file:///Users/aanyavarshney/Documents/GitHub/Neurovox/docs/ARCHITECTURE.md).

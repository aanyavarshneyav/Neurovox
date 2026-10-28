# PROJECT_SPEC.md — Consolidated Project Specification

**Title:** Scaled Reproduction of SKA-HNN and IAAMSoftmax for Text-Independent Speaker Verification on LibriSpeech  
**Reference Paper:** J. Alam, A. Fathan, M. S. Alam, *"A Hybrid Neural Approach to Speaker Verification with an Improved Additive Angular Margin Loss"*, IJCNN 2025, DOI 10.1109/IJCNN64981.2025.11228198.  
**Compute:** Apple Mac M3 Pro (MPS/CPU) primary; Google Colab secondary.

**Provenance legend.** Every claim carries exactly one tag:
- `[Paper]` — stated in the paper text, Table II/VIII, Algorithm 1, or Appendix A. Verified against the extracted PDF text on 2026-10-28.
- `[External-UNVERIFIED]` — taken from the PRD or companion work; not in the paper; tracked in [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).
- `[Engineering Choice]` — our decision; logged in [DECISIONS.md](DECISIONS.md).
- `[Derived]` — arithmetic from `[Paper]` dimensions (shows its inputs).

---

## 1. System Overview

SKA-HNN is a 2D-CNN (fwSKA + cwSKA ResBlocks) → TDNN-LSTM → Fully-TDNN cascade, with MLFC + MLASP pooling and two FC layers feeding IAAMSoftmax `[Paper, Sec. II, Fig. 1]`.

```
MFB (80-d, SpecAugment on the fly)             [Paper]
  → 2D-CNN module (Conv2d, 2×ResBlock, Conv2d)   [Paper]  → X_c ∈ R^{d_c×T}
  → [X_c; LS(X_c)] → TDNN-LSTM (1 TDNN + 1 LSTM)  [Paper]  → X_l ∈ R^{d_l×T}
  → [X_l; LS(X_l)] → Fully-TDNN (3 TDNN layers)   [Paper]  → X_t ∈ R^{d_t×T}
  → MLFC: X_i = [X_c; X_l; X_t]                   [Paper]
  → X_i^c = [X_i; μ_i; σ_i]                       [Paper]
  → MLASP (attention over X_i^c, reduced to X_i dim) → mean+std  [Paper]
  → FC+BN → FC+BN → IAAMSoftmax                   [Paper]
Embeddings: FC layer closest to pooling (192-d)   [Paper, Sec. II-D / Fig. 3 caption]
```

### 1.1 Acoustic front-end and augmentation

| Item | Value | Tag |
|---|---|---|
| Features | 80-d Mel filterbank | `[Paper]` |
| Window / shift | 25 ms / 10 ms | `[Paper]` |
| Window function | Hamming | `[Engineering Choice]` |
| Training segment | 180 frames | `[Paper]` |
| Augmentation set | MUSAN noise, speed 0.95–1.05, reverberation, SpecAugment | `[Paper]` |
| Reverb source | RIRS_NOISES (1.3 GB) | `[Engineering Choice]` (paper says "reverberation" only) |
| MUSAN replacement | On-the-fly in-batch babble + white/pink noise | `[Engineering Choice]` (ADR-006) |
| SpecAugment mask widths (F=10, T=15) | — | `[External-UNVERIFIED]` (PRD; paper gives no widths) |
| Noise SNR ranges, application probabilities | see DATA_PROTOCOL | `[Engineering Choice]` |

### 1.2 Module topology

| Module | Specification | Tag |
|---|---|---|
| 2D-CNN layout | Conv2d+ReLU+BN, 2× ResBlock, Conv2d+ReLU+BN; ResBlock = Conv2d, fwSKA, cwSKA, SE, skip | `[Paper]` |
| 2D-CNN shapes | F×T → C×F/2×T → … → C·F/4×T = X_c | `[Paper, Table VIII]` |
| Kernel sizes / strides | 3×3, stride (2,1) in the two outer Conv2d | `[External-UNVERIFIED]` |
| SKA branch count n, kernels, dilation | n=2; 3×3 (d=1) and 3×3 (d=2) | `[External-UNVERIFIED]` (paper: "n branches… pre-defined different kernel sizes") |
| SKA reduction ratio, SE ratio | r=8 | `[External-UNVERIFIED]` |
| Local statistics | 1st/2nd-order stats; "resampling… adjusting window shift" | `[Paper]` |
| LS window length | 21 frames, stride 1 | `[External-UNVERIFIED]` |
| LS padding | replicate | `[Engineering Choice]` |
| LS numerics | fp32; clamp var ≥ 1e-5 before sqrt | `[Engineering Choice]` |
| TDNN-LSTM | one TDNN + one LSTM; input [X_c; LS] = 3·C·F/4 | `[Paper]` |
| TDNN kernel/dilation, LSTM unidirectional | k=3 d=1; unidirectional | `[External-UNVERIFIED]` |
| Fully-TDNN | three TDNN layers; input 3·C_1 | `[Paper]` |
| Fully-TDNN dilations | d=1,2,3 | `[External-UNVERIFIED]` |
| MLFC | d_i = d_c + d_l + d_t | `[Paper]` |
| MLASP input / output | attention over 3·d_i, reduced to d_i; weighted mean+std → 2·d_i; BN | `[Paper, Sec. II-C, Table VIII]` |
| Attention bottleneck d_attn, activation | tanh, d_attn=256 full / 128 scaled | `[Engineering Choice]` (paper gives no d_attn; PRD value 256 `[External-UNVERIFIED]`) |
| Classifier | 2× (FC+BN), 192-d | `[Paper]` |
| Embedding layer | FC closest to pooling (FC1) | `[Paper]` |
| Extra diagnostic: also score FC2 | — | `[Engineering Choice]` (ADR-004) |

### 1.3 Width scaling

| Quantity | Full `[Paper, Table VIII / Sec. II-B]` | Scaled `[Engineering Choice]` |
|---|---|---|
| C | 128 | 32 |
| d_c = C·F/4 (F=80) | 2560 `[Paper]` | 640 `[Derived: 32×20]` |
| C_1 (= d_l = d_t) | 512 `[Paper]` | 256 |
| d_i | 3584 `[Paper]` | 1152 `[Derived: 640+256+256]` |
| MLASP output 2·d_i | 7168 `[Derived]` | 2304 `[Derived]` |
| Embedding dim | 192 `[Paper]` | 192 |
| Total parameters | 30.91M `[Paper, Table II]` | TBD: measure with torchinfo |

---

## 2. Loss: AAMSoftmax and IAAMSoftmax (AdaptiveDrop)

### 2.1 AAMSoftmax
$$\mathcal{L}=-\frac1N\sum_{i=1}^{N}\log\frac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)}+\sum_{j=1,\,j\ne y_i}^{C}e^{s\cos\theta_j}}$$
- s = 30, m = 0.2 `[Paper, Sec. IV-A]`.
- The paper's Eq. (3) writes the sum as $j\ne i$; this is an index typo, and we use $j\ne y_i$ `[Engineering Choice]` (ADR-001).

### 2.2 AdaptiveDrop (paper Algorithm 1)
1. Cosine similarity CS(x,y) between the on-the-fly embedding and the class center ω_y `[Paper]`.
2. Per step: if epoch ≥ ESC, correct labels; if epoch ≥ ESD, filter by CS vs. τ; else keep all `[Paper, Alg. 1]`.
3. τ = 0.423 (65°); filtering starts after a 5-epoch warm-up `[Paper, Sec. III]`.
4. Filter comparator: Sec. II text says keep if CS ≥ τ; Algorithm 1 says CS > τ. Immaterial for float scores; we use ≥ `[Engineering Choice]`.
5. **ESC value and the correction rule are not given in the paper.** Our choice: ESC = ESD = 5 (correction is applied before filtering within the same step, matching the paper's stated order) `[Engineering Choice]`. Correction rule: if CS(x,y) < τ and max_{c≠y} CS(x,c) ≥ τ_corr = 0.5, set ŷ = argmax_c CS(x,c) `[External-UNVERIFIED]` (PRD attributing it to companion work; see OPEN_QUESTIONS Q1).
6. Safety guard: if fewer than `min_keep_ratio·N` samples survive, keep the top-k by CS. Default `min_keep_ratio`=0.5, and this guard is itself an ablation switch `[Engineering Choice]` (ADR-001).

---

## 3. Training recipe

| Item | Value | Tag |
|---|---|---|
| Optimizer | Adam | `[Paper]` |
| Adam betas/eps | PyTorch defaults | `[Engineering Choice]` |
| LR schedule | initial 1e-7, triangular cyclic | `[Paper]` |
| LR peak (max_lr), cycle length | 1e-3; 2 epochs | `[Engineering Choice]` |
| Batch | 64 randomly selected speakers (1 segment each) | `[Paper]` |
| Weight decay, grad clip | 1e-4; 3.0 | `[Engineering Choice]` |
| Epochs | 30 initial; re-scoped after benchmark | `[Engineering Choice]` |
| Precision | fp32 on MPS; no CUDA GradScaler | `[Engineering Choice]` |

---

## 4. Evaluation protocol

| Item | Value | Tag |
|---|---|---|
| Scoring | cosine | `[Paper]` |
| Metrics | EER and minDCF | `[Paper]` |
| minDCF prior | paper does not state P_target; we report 0.01 and 0.05 (C_miss=C_fa=1) | `[Engineering Choice]` |
| Seeds | ≥ 3 for headline comparisons | `[Engineering Choice]` |
| CI | paired bootstrap over speakers/utterances, identical resamples across systems | `[Engineering Choice]` |
| Target trials | 100% cross-chapter; single-chapter speakers discarded (and their count logged) | `[Engineering Choice]` |
| Impostor trials | same-gender | `[Engineering Choice]` |
| test-other | acoustic-robustness set, not an analogue of VoxCeleb1-H (which tests hard impostor pairs) | `[Engineering Choice]` |

Minimum DCF: $C_{det}(\theta)=C_{miss}P_{miss}(\theta)P_{tar}+C_{fa}P_{fa}(\theta)(1-P_{tar})$, normalized by $\min(C_{miss}P_{tar},\,C_{fa}(1-P_{tar}))$, minimized over θ `[Engineering Choice]`.

---

## 5. Dataset rationale (LibriSpeech vs LibriTTS)

LibriSpeech is native 16 kHz, matching the paper's 80-Mel/25 ms/10 ms front-end, and retains natural pauses. LibriTTS is 24 kHz and trimmed for TTS. A 24 kHz rate does *not* distort the Mel axis, since 80 Mels up to 8 kHz can be computed at any rate. The choice rests on cost and on preserved pauses only `[Engineering Choice]`.

---

## 6. Demo guidance (Tier 4, optional)

- Decision thresholds are calibrated from validation DET/score distributions, with no promised cosine ranges `[Engineering Choice]`.
- MLASP attention is per-channel, so frame-level plots average over channels `[Engineering Choice]`.

---

## 7. References (corrected attributions)

- ASP: Okabe et al., Interspeech 2018 `[External-UNVERIFIED]`. Channel-dependent ASP: Desplanques et al., Interspeech 2020 `[External-UNVERIFIED]`.
- SKA-TDNN: Mun et al., IEEE SLT 2023 `[Paper, ref. 24]`.
- SKNet: Li et al., CVPR 2019 `[Paper, ref. 36]`. ArcFace: Deng et al., CVPR 2019 `[Paper, ref. 38]`. AdaptiveDrop: Fathan, Zhu, Alam, ICASSP 2025 `[Paper, ref. 39]`.
- The `readourlips/SKA-TDNN` repository URL is unverified, so we do not use it (OPEN_QUESTIONS Q5).

---

## 8. Changes from the PRD

| Area | PRD | Now | Source |
|---|---|---|---|
| Parameter counts | hand-computed (MLASP 8.81M with d_attn=614; ResBlock 1.334M; scaled 2.65M; S6 1.54M) | paper Table II numbers for the full model; all others "TBD: measure with torchinfo" | PRD_review #1 |
| Baseline | Tier 0 at 1.95M vs. a larger scaled model | both a matched and a standard-width baseline, side by side, with measured counts | PRD_review #1 |
| Expected results / gates | stated as facts; gates required improvement | all expectations labeled HYPOTHESIS; gates technical only | PRD_review #2 |
| Scaling-law "defense" | EER ≈ 9.15 − 1.08 ln N | removed; gap explained qualitatively | PRD_review #3 |
| SSL pseudo-labels | K-Means on bootstrap embeddings / MFCC stats | primary: synthetic noise 10/20/40%; secondary: K-Means on frozen WavLM-Base+; purity/NMI logged | PRD_review #4 |
| ESC/ESD | ESD=5, ESC=7 (filtering before correction) | ESC=ESD=5, correction before filtering | PRD_review #5, paper Alg. 1 |
| d_attn | 614 (reverse-engineered) | 256/128, tagged as our choice | PRD_review #1 |
| Embedding layer | FC2 | FC1 (paper) with FC2 as extra diagnostic | paper Sec. II-D |
| fp16 stats | allowed | fp32 + clamp | PRD_review moderate |
| Trial lists | silent same-chapter fallback; trial-level bootstrap | cross-chapter only; paired speaker/utterance bootstrap; ≥3 seeds | PRD_review moderate |
| minDCF prior | unspecified | both 0.01 and 0.05 | PRD_review moderate |
| LibriTTS argument | "distorts Mel axis" | removed | PRD_review moderate |
| ASP / SKA-TDNN citations | Desplanques 2020; SLT 2022 | Okabe 2018; SLT 2023 | PRD_review minor |

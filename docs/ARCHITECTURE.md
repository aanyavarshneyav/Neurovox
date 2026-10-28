# ARCHITECTURE.md — Module Contracts & Tensor Shapes

**Reference Architecture:** SKA-HNN (Selective Kernel Attention-driven Hybrid Neural Network)  
**Paper Reference:** Alam et al., IJCNN 2025  
**Scaling Scheme:** Scaled dimensions ($C=32, C_1=256$) for 251-speaker LibriSpeech reproduction; full paper dimensions ($C=128, C_1=512$) documented for reference.

---

## 1. Global Tensor Flow & Notation

- $B$: Mini-batch size (number of utterances; default $B=64$).
- $T$: Duration in frames (default $T=180$ frames during training, ~1.8 seconds at 10 ms frame shift).
- $F$: Acoustic Mel-Filterbank feature bins ($F=80$).
- $C$: Number of 2D-CNN channels (Full: $C=128$; Scaled: $C=32$).
- $C_1$: Number of TDNN/LSTM channels (Full: $C_1=512$; Scaled: $C_1=256$).
- $d_c$: Flattened 2D-CNN output dimension ($d_c = C \times (F / 4) = 20 \times C$; Full: 2560; Scaled: 640).
- $d_i$: Multi-Level Feature Combination dimension ($d_i = d_c + C_1 + C_1$; Full: 3584; Scaled: 1152).
- $d_{attn}$: MLASP attention bottleneck dimension (Full: 256; Scaled: 128).
- $N_c$: Number of training speaker classes ($N_c = 251$ for LibriSpeech train-clean-100; $N_c = 5994$ for VoxCeleb2 dev).

---

## 1b. Provenance of the contracts below

Tags follow [PROJECT_SPEC.md](PROJECT_SPEC.md). Shapes and dimensions come from Table VIII and Sec. II-B/C.

| Contract element | Tag |
| :--- | :--- |
| Layer order, shapes (C·F/2·T, C·F/4·T, 3·C·F/4, 3·C_1, 3584, 2·3584, 192, N_c), d_c=2560, d_l=d_t=512, d_i=3584 at C=128, C_1=512 | `[Paper]` |
| Conv kernel sizes, strides, paddings, dilations, LSTM unidirectional, LS window W=21 | `[External-UNVERIFIED]` |
| Replicate padding, fp32 + variance clamp in LS | `[Engineering Choice]` |
| MLASP bottleneck (Conv1d-Tanh-BN-Conv1d-Softmax) and d_attn | `[Engineering Choice]` (paper: weights over 3·d_i "reduced to match" d_i, no layer spec) |
| SKA internals in §3.1, SE ratio | `[External-UNVERIFIED]` |
| Scaled widths (C=32, C_1=256) | `[Engineering Choice]` |
| All parameter counts | TBD: measure with torchinfo (the only exceptions are Table II numbers, tagged `[Paper]`) |

---

## 2. Module-by-Module Tensor Contracts

| Stage | Module Name | Input Shape `(B, ...)` | Internal Layers & Operations | Output Shape `(B, ...)` | Parameter Count (Full Scale) | Parameter Count (Scaled Project) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0. Acoustic Frontend** | Mel-Filterbank + SpecAugment | `(B, 1, 80, T)` | 80 Mels, 25ms win, 10ms hop; time/freq masking, speed perturbation, synthetic noise / RIR | `(B, 1, 80, T)` | Non-parametric | Non-parametric |
| **1. 2D-CNN Frontend** | Preceding Conv2d Block | `(B, 1, 80, T)` | $\text{Conv2d}(1, C, k=3, s=(2,1), p=1) \to \text{ReLU} \to \text{BN2d}(C)$ | `(B, C, 40, T)` | TBD: measure with torchinfo | TBD: measure with torchinfo |
| **1. 2D-CNN Frontend** | SKA ResBlock 1 | `(B, C, 40, T)` | $\text{Conv2d}(C, C, 3, s=1, p=1) \to \text{fwSKA} \to \text{cwSKA} \to \text{SE}(r=8) + \text{Skip}$ | `(B, C, 40, T)` | TBD: measure with torchinfo | TBD: measure with torchinfo |
| **1. 2D-CNN Frontend** | SKA ResBlock 2 | `(B, C, 40, T)` | Identical topology to ResBlock 1 | `(B, C, 40, T)` | TBD: measure with torchinfo | TBD: measure with torchinfo |
| **1. 2D-CNN Frontend** | Succeeding Conv2d Block | `(B, C, 40, T)` | $\text{Conv2d}(C, C, k=3, s=(2,1), p=1) \to \text{ReLU} \to \text{BN2d}(C)$ | `(B, C, 20, T)` | TBD: measure with torchinfo | TBD: measure with torchinfo |
| **1. 2D-CNN Frontend** | Frequency-Channel Flatten | `(B, C, 20, T)` | Reshape / flatten frequency and channel dimensions: $d_c = 20 \times C$ | `(B, d_c, T)` | Non-parametric | Non-parametric |
| **2. TDNN-LSTM Module** | Local Statistics (LS) 1 | `(B, d_c, T)` | Sliding window ($W=21$) mean $\mu$ and std $\sigma$ in fp32 with replicate padding | `(B, 2*d_c, T)` | Non-parametric | Non-parametric |
| **2. TDNN-LSTM Module** | LS Concatenation 1 | `(B, d_c, T)` & `(B, 2*d_c, T)` | Concatenate $[X_c; \mu_c; \sigma_c]$ along channel dimension | `(B, 3*d_c, T)` | Non-parametric | Non-parametric |
| **2. TDNN-LSTM Module** | TDNN Layer | `(B, 3*d_c, T)` | $\text{Conv1d}(3d_c, C_1, k=3, p=1) \to \text{ReLU} \to \text{BN1d}(C_1)$ | `(B, C_1, T)` | TBD: measure with torchinfo | TBD: measure with torchinfo |
| **2. TDNN-LSTM Module** | LSTM Layer | `(B, C_1, T)` | Unidirectional $\text{LSTM}(C_1, C_1, \text{num\_layers}=1)$ | `(B, C_1, T)` | TBD: measure with torchinfo | TBD: measure with torchinfo |
| **3. Fully-TDNN Module** | Local Statistics (LS) 2 | `(B, C_1, T)` | Sliding window ($W=21$) mean $\mu$ and std $\sigma$ in fp32 with replicate padding | `(B, 2*C_1, T)` | Non-parametric | Non-parametric |
| **3. Fully-TDNN Module** | LS Concatenation 2 | `(B, C_1, T)` & `(B, 2*C_1, T)` | Concatenate $[X_l; \mu_l; \sigma_l]$ along channel dimension | `(B, 3*C_1, T)` | Non-parametric | Non-parametric |
| **3. Fully-TDNN Module** | TDNN Layer 1 | `(B, 3*C_1, T)` | $\text{Conv1d}(3C_1, C_1, k=3, d=1, p=1) \to \text{ReLU} \to \text{BN1d}(C_1)$ | `(B, C_1, T)` | TBD: measure with torchinfo | TBD: measure with torchinfo |
| **3. Fully-TDNN Module** | TDNN Layer 2 | `(B, C_1, T)` | $\text{Conv1d}(C_1, C_1, k=3, d=2, p=2) \to \text{ReLU} \to \text{BN1d}(C_1)$ | `(B, C_1, T)` | TBD: measure with torchinfo | TBD: measure with torchinfo |
| **3. Fully-TDNN Module** | TDNN Layer 3 | `(B, C_1, T)` | $\text{Conv1d}(C_1, C_1, k=3, d=3, p=3) \to \text{ReLU} \to \text{BN1d}(C_1)$ | `(B, C_1, T)` | TBD: measure with torchinfo | TBD: measure with torchinfo |
| **4. MLASP Module** | MLFC Concatenation | `(B, d_c, T)`, `(B, C_1, T)`, `(B, C_1, T)` | Concatenate $[X_c; X_l; X_t]$ along channel dimension: $d_i = d_c + 2C_1$ | `(B, d_i, T)` | Non-parametric | Non-parametric |
| **4. MLASP Module** | Local Statistics (LS) 3 | `(B, d_i, T)` | Sliding window ($W=21$) mean $\mu$ and std $\sigma$ in fp32 with replicate padding | `(B, 2*d_i, T)` | Non-parametric | Non-parametric |
| **4. MLASP Module** | Context Concatenation | `(B, d_i, T)` & `(B, 2*d_i, T)` | Concatenate $[X_i; \mu_i; \sigma_i]$ to form context tensor $X_i^c$ | `(B, 3*d_i, T)` | Non-parametric | Non-parametric |
| **4. MLASP Module** | Channel Attention Scoring | `(B, 3*d_i, T)` | $\text{Conv1d}(3d_i, d_{attn}, 1) \to \text{Tanh} \to \text{BN1d}(d_{attn}) \to \text{Conv1d}(d_{attn}, d_i, 1) \to \text{Softmax}(T)$ | `(B, d_i, T)` | TBD: measure with torchinfo | TBD: measure with torchinfo |
| **4. MLASP Module** | Weighted Stats Pooling | `(B, d_i, T)` & weights | Compute weighted mean $\tilde{\mu} = \sum_t \alpha_t x_t$ and weighted std $\tilde{\sigma} = \sqrt{\sum_t \alpha_t x_t^2 - \tilde{\mu}^2}$ | `(B, 2*d_i)` | Non-parametric | Non-parametric |
| **4. MLASP Module** | Pooling BatchNorm | `(B, 2*d_i)` | $\text{BN1d}(2d_i)$ applied to pooled utterance representation | `(B, 2*d_i)` | TBD: measure with torchinfo | TBD: measure with torchinfo |
| **5. Classifier / Head** | FC Layer 1 (primary embedding, `[Paper]`) | `(B, 2*d_i)` | $\text{Linear}(2d_i, 192) \to \text{BN1d}(192)$; paper: embeddings come from the FC layer closest to pooling | `(B, 192)` | TBD: measure with torchinfo | TBD: measure with torchinfo |
| **5. Classifier / Head** | FC Layer 2 (diagnostic embedding, `[Engineering Choice]`) | `(B, 192)` | $\text{Linear}(192, 192) \to \text{BN1d}(192)$ | `(B, 192)` | TBD: measure with torchinfo | TBD: measure with torchinfo |
| **5. Classifier / Head** | IAAMSoftmax Margin Loss | `(B, 192)` & labels | ArcFace margin projection ($s=30, m=0.2$) with AdaptiveDrop filtering ($\tau=0.423$) | `(B, N_c)` logits & scalar loss | TBD: measure with torchinfo | TBD: measure with torchinfo |

---

## 3. Sub-Component Architectural Details

### 3.1 Frequency-wise and Channel-wise Selective Kernel Attention (fwSKA & cwSKA)
The paper states only that each module does Split (n branches, different kernel sizes), Fuse (element-wise) and Select (soft attention, frequency-wise or channel-wise) `[Paper]`. The concrete branch count, kernels, pooling axes and MLP sizes below are `[External-UNVERIFIED]` / `[Engineering Choice]` and may change after verification against the SKA-TDNN paper:
1. **Split:**
   - Branch 1: $\text{Conv2d}(C_{in}, C_{in}, k=3, p=1, d=1)$
   - Branch 2: $\text{Conv2d}(C_{in}, C_{in}, k=3, p=2, d=2)$ (dilation 2 achieves a $5\times 5$ receptive field with identical parameter count)
2. **Fuse:**
   - Multi-scale features are summed: $U = U_1 + U_2 \in \mathbb{R}^{B \times C \times F \times T}$.
3. **Select (fwSKA vs. cwSKA):**
   - **fwSKA (Frequency-wise):** Global average pooling is applied across time and channel axes:
     $$s_{freq} = \frac{1}{C \times T} \sum_{c=1}^C \sum_{t=1}^T U[:, c, :, t] \in \mathbb{R}^{B \times F}$$
     Attention vector is produced via a bottleneck MLP with reduction ratio $r$ (default 8, `[External-UNVERIFIED]`) followed by a softmax across branches.
   - **cwSKA (Channel-wise):** Global average pooling is applied across time and frequency axes:
     $$s_{chan} = \frac{1}{F \times T} \sum_{f=1}^F \sum_{t=1}^T U[:, :, f, t] \in \mathbb{R}^{B \times C}$$
     Attention vector is produced via a bottleneck MLP with the same reduction ratio $r$ followed by a softmax across branches (`[External-UNVERIFIED]`).
   - The attended output is the soft-weighted combination: $V = a_1 \odot U_1 + a_2 \odot U_2$.

### 3.2 Sliding-Window Local Statistics Pooling
Given a sequence tensor $X \in \mathbb{R}^{B \times D \times T}$:
1. Cast input to `torch.float32`.
2. Pad temporal boundaries with replication: `pad = W // 2` (where $W=21$).
3. Compute moving mean: $\mu_t = \text{avg\_pool1d}(X_{padded}, \text{kernel\_size}=W, \text{stride}=1)$.
4. Compute moving second moment: $\mu^{(2)}_t = \text{avg\_pool1d}(X^2_{padded}, \text{kernel\_size}=W, \text{stride}=1)$.
5. Compute variance: $\text{var}_t = \mu^{(2)}_t - \mu_t^2$.
6. Clamp variance: $\text{var}_{clamped} = \text{torch.clamp}(\text{var}_t, \text{min}=10^{-5})$.
7. Standard deviation: $\sigma_t = \sqrt{\text{var}_{clamped}}$.
8. Output tensor: $[X; \mu; \sigma] \in \mathbb{R}^{B \times 3D \times T}$.

---

## 4. Mapping of Ablation Switches (S0–S6)

Definitions S0–S6 and the parameter column are taken from Table II and the Sec. IV-B text of the paper `[Paper]` (verified against the extracted PDF text). Scaled-model parameter counts are not in the paper and are measured here.

| Switch ID | Architecture & Loss Configuration | Config Flag Mapping | Paper Description | Paper Params `[Paper, Table II]` | Measured Params (scaled) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S0** | **Full Proposed System:** SKA-HNN + IAAMSoftmax | `loss: iaam_softmax`<br>`mlfc: true`<br>`mlasp_context: true`<br>`frontend: ska_2dcnn` | Full framework with 2D-CNN, MLFC, MLASP context, and IAAMSoftmax. | 30.91M | TBD: measure with torchinfo |
| **S1** | **AAMSoftmax Only:** SKA-HNN + standard AAMSoftmax | `loss: aam_softmax`<br>`mlfc: true`<br>`mlasp_context: true`<br>`frontend: ska_2dcnn` | Same as S0, but utilizing standard AAMSoftmax (no AdaptiveDrop filtering/correction). | 30.91M | TBD: measure with torchinfo |
| **S2** | **No MLASP Context:** S1 without context in MLASP | `loss: aam_softmax`<br>`mlfc: true`<br>`mlasp_context: false`<br>`frontend: ska_2dcnn` | Attention weights computed directly over $X_i \in \mathbb{R}^{d_i \times T}$ instead of $X_i^c \in \mathbb{R}^{3d_i \times T}$. | 29.16M | TBD: measure with torchinfo |
| **S3** | **No MLFC:** S1 without multi-level feature combination | `loss: aam_softmax`<br>`mlfc: false`<br>`mlasp_context: true`<br>`frontend: ska_2dcnn` | $X_i = X_t \in \mathbb{R}^{C_1 \times T}$. MLASP receives only Fully-TDNN output and its local statistics. | 26.77M | TBD: measure with torchinfo |
| **S4** | **No MLFC & No Context:** S1 without MLFC and without context | `loss: aam_softmax`<br>`mlfc: false`<br>`mlasp_context: false`<br>`frontend: ska_2dcnn` | $X_i = X_t \in \mathbb{R}^{C_1 \times T}$. MLASP applied directly on $X_t$ without local statistics. | 26.52M | TBD: measure with torchinfo |
| **S5** | **Vanilla 2D-CNN:** S1 with standard Conv2d (no SKA) | `loss: aam_softmax`<br>`mlfc: true`<br>`mlasp_context: true`<br>`frontend: vanilla_2dcnn` | Replaces fwSKA + cwSKA inside ResBlocks with standard single-branch 2D convolutions. | 17.37M | TBD: measure with torchinfo |
| **S6** | **No 2D-CNN Frontend:** S1 without any 2D-CNN module | `loss: aam_softmax`<br>`mlfc: true`<br>`mlasp_context: true`<br>`frontend: none` | Acoustic features feed directly into TDNN-LSTM. MLFC pools over $X_i = [X_l; X_t]$. | 6.34M | TBD: measure with torchinfo |

### 4.1 Note on Paper Parameter Jumps
> [!NOTE]
> Table II gives S1 = 30.91M and S5 = 17.37M, a 13.54M difference `[Derived]`. PRD_review argues that replacing SKA with vanilla convs in the paper's ResBlocks cannot explain a delta that large `[External-UNVERIFIED: reviewer estimate]`. Either the paper's front-end differs from this reconstruction or S5 changes widths. This is tracked as OPEN_QUESTIONS Q2, and all parameter numbers will be measured with `torchinfo` and logged. The paper itself remarks that reduced complexity lowers parameters but degrades performance `[Paper, Sec. IV-B]`; the ablations vary parameters and structure together.

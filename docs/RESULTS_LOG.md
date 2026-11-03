# RESULTS_LOG.md — Experimental Run Ledger

**Integrity Rule:** No number may be cited as an experimental result unless it is logged in this ledger with an active `run_id`, configuration hash, seed, and measured metrics. Anticipated or literature numbers are strictly labeled `HYPOTHESIS`.

---

## 1. Experimental Ledger

| Run ID | Timestamp | Config Hash | Seed | Model Architecture | Loss Function | Dataset / Condition | Measured Params (`torchinfo`) | EER (%) | minDCF ($P=0.01$) | minDCF ($P=0.05$) | Device / Hardware | Run Status | Notes & Hypotheses |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *Template* | *YYYY-MM-DD* | *sha256* | *42* | *scaled_ska_hnn* | *iaam_softmax* | *LibriSpeech test-clean* | *TBD: measure* | *TBD* | *TBD* | *TBD* | *Mac M3 Pro (mps)* | *PENDING* | *Initial run template* |
| `ref_speechbrain_ecapa_test_clean` | 2026-10-31 | `spkrec-ecapa-voxceleb` | N/A (pretrained) | ECAPA-TDNN (SpeechBrain $C=1024$) | AAM-Softmax (pretrained VoxCeleb1+2) | LibriSpeech test-clean (5,800 trials, SHA-256: `371a42e7...`) | 20,767,552 (20.77M) | 2.52% | 0.0016 (raw) | 0.0053 (raw) | Mac M3 Pro (`device = mps`, fp32) | SUPERSEDED (v1 trials) | Superseded by v2 trial list (ADR-017). Evaluated on unnormalized minDCF without speaker CIs. |
| `ref_speechbrain_ecapa_test_other` | 2026-10-31 | `spkrec-ecapa-voxceleb` | N/A (pretrained) | ECAPA-TDNN (SpeechBrain $C=1024$) | AAM-Softmax (pretrained VoxCeleb1+2) | LibriSpeech test-other (5,800 trials, SHA-256: `fa3b756a...`) | 20,767,552 (20.77M) | 1.79% | 0.0018 (raw) | 0.0054 (raw) | Mac M3 Pro (`device = mps`, fp32) | SUPERSEDED (v1 trials) | Superseded by v2 trial list (ADR-017). Evaluated on unnormalized minDCF without speaker CIs. |
| `ref_speechbrain_ecapa_test_clean_v2` | 2026-11-03 | `spkrec-ecapa-voxceleb` | N/A (pretrained) | ECAPA-TDNN (SpeechBrain $C=1024$) | AAM-Softmax (pretrained VoxCeleb1+2) | LibriSpeech test-clean v2 (5,800 trials, SHA-256: `5b47db5e...`) | 20,767,552 (20.77M) | 2.41% [1.31%, 3.43%] | norm: 0.1183 [0.0379, 0.1604] (raw: 0.0012) | norm: 0.0934 [0.0365, 0.1357] (raw: 0.0047) | Mac M3 Pro (`device = mps`, fp32) | COMPLETED [EXTERNAL REFERENCE] | Sanity check external reference on v2 trial list; 95% speaker bootstrap CI over 1,000 iterations. |
| `ref_speechbrain_ecapa_test_other_v2` | 2026-11-03 | `spkrec-ecapa-voxceleb` | N/A (pretrained) | ECAPA-TDNN (SpeechBrain $C=1024$) | AAM-Softmax (pretrained VoxCeleb1+2) | LibriSpeech test-other v2 (5,800 trials, SHA-256: `92e651bc...`) | 20,767,552 (20.77M) | 1.72% [1.05%, 2.47%] | norm: 0.0969 [0.0314, 0.1286] (raw: 0.0010) | norm: 0.0648 [0.0314, 0.0954] (raw: 0.0032) | Mac M3 Pro (`device = mps`, fp32) | COMPLETED [EXTERNAL REFERENCE] | Sanity check external reference on v2 trial list; 95% speaker bootstrap CI over 1,000 iterations. |

---

## 2. Experimental Entries

### Pretrained SpeechBrain ECAPA-TDNN Benchmark v2 (External Reference)
- **Run ID:** `ref_speechbrain_ecapa_test_clean_v2` & `ref_speechbrain_ecapa_test_other_v2`
- **Timestamp:** 2026-11-03
- **Platform:** Apple Silicon MacBook Pro 14-inch (Apple M3 Pro, 11 cores, 18 GB RAM, macOS 26.5 Build 25F71)
- **Device & Precision:** `device = mps`, `torch.float32` (PYTORCH_ENABLE_MPS_FALLBACK=1)
- **Software:** Python 3.12.5, PyTorch 2.15.0, TorchAudio 2.11.1, SpeechBrain 1.1.1, SoundFile 0.14.0, TorchInfo 1.8.0
- **Model Checkpoint:** `speechbrain/spkrec-ecapa-voxceleb` (pre-trained on VoxCeleb 1 & 2, standard width $C=1024$)
- **Measured Parameters (`torchinfo`):** 20,767,552 parameters (20.77M)
- **Trial Lists (v2):**
  - `trials_test-clean.txt`: 5,800 trials (2,900 cross-chapter targets, 2,900 gender-matched impostors, 100 targets/speaker; SHA-256: `5b47db5e847990a75b35edfc5d906944dd069a59144df7c6b5f5126388ee6aa1`)
  - `trials_test-other.txt`: 5,800 trials (2,900 cross-chapter targets, 2,900 gender-matched impostors, 100 targets/speaker; SHA-256: `92e651bccc3af6253b6c97437d3197812cf384490cbe16cd812253779651a9a6`)
- **Measured Metrics (Point Estimate & 95% Speaker-Level Clustered Bootstrap CI, 1,000 iterations):**
  - `test-clean`:
    - **EER:** **2.41%** [95% CI: 1.31%, 3.43%] (threshold: 0.3221)
    - **minDCF ($P=0.05$, primary, norm):** **0.0934** [95% CI: 0.0365, 0.1357] (raw: 0.0047)
    - **minDCF ($P=0.01$, indicative, norm):** **0.1183** [95% CI: 0.0379, 0.1604] (raw: 0.0012)
  - `test-other`:
    - **EER:** **1.72%** [95% CI: 1.05%, 2.47%] (threshold: 0.3331)
    - **minDCF ($P=0.05$, primary, norm):** **0.0648** [95% CI: 0.0314, 0.0954] (raw: 0.0032)
    - **minDCF ($P=0.01$, indicative, norm):** **0.0969** [95% CI: 0.0314, 0.1286] (raw: 0.0010)
  - **Two-Sample Independent Speaker Bootstrap Difference (`test-other` minus `test-clean`):**
    - $\Delta \text{EER} = \mathbf{-0.69\%}$ [95% CI: $-1.88\%$, $+0.73\%$]
    - $\Delta \text{minDCF}_{0.05} = \mathbf{-0.0286}$ [95% CI: $-0.0859$, $+0.0328$]
    - $\Delta \text{minDCF}_{0.01} = \mathbf{-0.0214}$ [95% CI: $-0.1059$, $+0.0628$]
- **Status:** COMPLETED [EXTERNAL REFERENCE]
- **Diagnostic Findings & Methodological Interpretation:**
  1. *Plausibility Reference:* The measured EERs fall within standard biometric bounds ($0.1\% \le \text{EER} \le 10.0\%$), serving as an external reference that verification scoring mechanics and feature pipelines behave predictably. An external model's plausible performance is a sanity check confirming system integration, not mathematical proof of zero channel leakage or shortcuts in data manifests.
  2. *Evaluation of `test-other` vs `test-clean`:* The point-estimate EER for `test-other` (1.72%) is lower than `test-clean` (2.41%). However, the 95% confidence interval for their difference spans zero ($\Delta \text{EER} \in [-1.88\%, +0.73\%]$), demonstrating that the difference is not statistically significant. Candidate explanatory hypotheses include:
     - *Speaker sampling variance:* With only 29 qualifying speakers per test set, individual speaker difficulty dominates cohort metrics.
     - *Acoustic robustness of VoxCeleb ECAPA:* Pretrained on in-the-wild audio, the network may exhibit robust representation against varied recording environments.
     - *Cohort acoustic characteristics:* Variations in pitch, speaking rate, or accent among the specific 29 speakers selected for each set.
  3. *Statistical Power & Detectable Differences:* Because each test set yields ~29 qualifying multi-chapter speakers, the 95% confidence interval width on EER spans approximately $\pm 1.0\%$. Consequently, fine-grained differences between competing architectures cannot be declared meaningful from single-seed point estimates alone; multi-seed runs ($\ge 3$ seeds) and paired speaker-bootstrap difference CIs are strictly required.
- **Partition Demographics & Excluded Speakers (Audit Verified):**
  - `test-clean`: 40 total speakers (20 F / 20 M). 11 single-chapter speakers excluded (1188, 2094, 2300, 3575, 3729, 4507, 5639, 672, 7729, 8230, 8455). Qualifying cohort: 29 speakers (16 F / 13 M).
  - `test-other`: 33 total speakers in manifest (17 F / 16 M). 4 single-chapter speakers excluded (1688, 5764, 6938, 8280). Qualifying cohort: 29 speakers (15 F / 14 M).
  - `dev-clean`: 40 total speakers (20 F / 20 M). 9 single-chapter speakers excluded (1919, 2078, 3000, 3081, 3170, 3576, 3853, 422, 777). Qualifying cohort: 31 speakers (16 F / 15 M).

---

### Pretrained SpeechBrain ECAPA-TDNN Benchmark v1 (SUPERSEDED)
- **Run ID:** `ref_speechbrain_ecapa_test_clean` & `ref_speechbrain_ecapa_test_other`
- **Timestamp:** 2026-10-31
- **Status:** SUPERSEDED (Evaluated on v1 trial lists prior to H4/H5 fixes; superseded by ADR-017 and v2 runs above).

---

### Hardware & Throughput Benchmarks
- **Run ID:** `bench_m3pro_01`
- **Timestamp:** 2026-10-31
- **Platform:** Apple Silicon MacBook Pro 14-inch (Apple M3 Pro, 11 cores, 18 GB RAM, macOS 26.5 Build 25F71, Kernel Darwin 25.5.0)
- **Device & Precision:** `device = mps`, `torch.float32` (autocast fp16/bf16 evaluated and documented in `docs/BENCHMARKS.md`; fp32 confirmed default)
- **Software:** Python 3.12.5, PyTorch 2.15.0, TorchAudio 2.11.1, SoundFile 0.14.0
- **Measured Metrics:**
  - Conv2d Stack Plain ($C=32$, 4 plain convs, lower bound): 58.78 ms/step (17.01 steps/s)
  - Conv2d Stack with SKA ($C=32$, fwSKA + cwSKA + SE + Skip): 282.78 ms/step (3.54 steps/s)
  - nn.LSTM ($C_1=256$): 21.17 ms/step (47.25 steps/s)
  - Conv1d Scaled ($1920 \to 256$): 26.17 ms/step (38.22 steps/s)
  - Composite Pipeline Lower Bound ($C=32, C_1=256$, B=64): 116.51 ms/step (8.58 steps/s), peak driver memory 2,512.03 MB
  - Composite Pipeline Full SKA ($C=32, C_1=256$, B=64): 411.52 ms/step (2.43 steps/s), peak driver memory 4,568.25 MB
  - DataLoader Throughput (CPU, 4 workers, LibriSpeech dev-clean): 2,428.12 samples/s (26.36 ms/batch)
- **Status:** COMPLETED (Passed technical gate)
- **Derived Estimates:** Full SKA model ~201.9 s/epoch (446 batches of $B=64$); ~100.9 min per 30-epoch run; ~14.3 runs/day capacity; full 51-run portfolio ~85.8 hours (~3.58 days); pruned 31-run portfolio ~52.2 hours (~2.17 days). Must be re-measured after the first real training epoch.

*(Model evaluation entries will be appended chronologically as runs complete via `.agents/workflows/new-experiment.md`.)*

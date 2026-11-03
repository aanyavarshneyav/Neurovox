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
| `sanity_2a_overfit` | 2026-11-03 | `88b26f9e` | 42 | `small_tdnn_se` | AAMSoftmax ($s=30, m=0.2$) | 1 fixed batch (64 spk) | 410,928 (0.411M) | N/A (overfit) | N/A | N/A | Mac M3 Pro (`device = mps`, fp32) | SUPERSEDED (v2 rerun) | Train acc: 100.0%, Loss: 13.3750 -> 0.000011 (1,207,242x reduction), 0 NaNs. Constant LR 1e-3, 300 steps. Superseded by sanity_2a_overfit_v2. |
| `sanity_2a_50spk` | 2026-11-03 | `d3d0115b` | 42 | `small_tdnn_se` | AAMSoftmax (num_classes=50, $s=30, m=0.2$) | 50-speaker subset (5,695 utts) | 410,928 (0.411M) | 20.58% (best dev ep 2) | N/A | 0.8055 (best dev norm ep 2) | Mac M3 Pro (`device = mps`, fp32) | SUPERSEDED (set_epoch bug) | 3 epochs, B=32. Loss: 8.709 -> 5.184 -> 3.610. Acc: 26.0% -> 65.2% -> 78.1%. Superseded by sanity_2a_50spk_v2 due to batch sampler epoch omission. |
| `sanity_2a_full` | 2026-11-03 | `70443f48` | 42 | `small_tdnn_se` | AAMSoftmax (num_classes=251, $s=30, m=0.2$) | LibriSpeech train-clean-100 (251 spk, 28,539 utts) | 410,928 (0.411M) | 10.71% (best dev ep 10) | N/A | 0.4539 (best dev norm ep 10) | Mac M3 Pro (`device = mps`, fp32) | SUPERSEDED (set_epoch bug) | 10 epochs, B=64. Train-probe EER: 16.44% -> 3.90%. Dev EER: 23.84% -> 10.71%. Superseded by sanity_2a_full_v2 (only 63.2% unique utts visited due to set_epoch bug). |
| `sanity_2a_overfit_v2` | 2026-11-03 | `a7c9b39a` | 42 | `small_tdnn_se` | AAMSoftmax ($s=30, m=0.2$) | 1 fixed batch (64 spk) | 410,928 (0.411M) | N/A (overfit) | N/A | N/A | Mac M3 Pro (`device = mps`, fp32) | PROVISIONAL (superseded by v3) | Train acc: 100.0%, Loss: 13.5060 -> 0.000012 (1,128,742.8x reduction), 0 NaNs. Superseded by sanity_2a_overfit_v3 (ADR-020.2). |
| `sanity_2a_50spk_v2` | 2026-11-03 | `d3d0115b` | 42 | `small_tdnn_se` | AAMSoftmax (num_classes=50, $s=30, m=0.2$) | 50-speaker subset (5,695 utts) | 410,928 (0.411M) | 20.39% (best dev ep 2) | N/A | 0.7813 (best dev norm ep 2) | Mac M3 Pro (`device = mps`, fp32) | PROVISIONAL (superseded by v3) | 3 epochs, B=32. Step-0 loss: 11.5149. Dev EER: 34.58% (ep 0) -> 20.39% (ep 2). Superseded by sanity_2a_50spk_v3 (ADR-020.2). |
| `sanity_2a_full_v2` | 2026-11-03 | `70443f48` | 42 | `small_tdnn_se` | AAMSoftmax (num_classes=251, $s=30, m=0.2$) | LibriSpeech train-clean-100 (251 spk, 28,539 utts) | 410,928 (0.411M) | 10.77% (best dev ep 10) | N/A | 0.4548 (best dev norm ep 8) | Mac M3 Pro (`device = mps`, fp32) | PROVISIONAL (superseded by v3) | 10 epochs, B=64, 446 b/ep. Step-0 loss: 13.6501. Dev EER: 34.58% -> 10.77%. Superseded by sanity_2a_full_v3 (ADR-020.2). |
| `sanity_2a_overfit_v3` | 2026-11-03 | `88b26f9e` | 42 | `small_tdnn_se` | AAMSoftmax ($s=30, m=0.2$) | 1 fixed batch (64 spk) | 410,928 (0.411M) | N/A (overfit) | N/A | N/A | Mac M3 Pro (`device = mps`, fp32) | COMPLETED [GATE PASSED] | Commit `33499df`. Train acc: 100.0%, Loss: 13.4322 -> 0.000012 (1,119,347.3x reduction), 0 NaNs. Constant LR 1e-3, 300 steps. Verified with draw_id 3-tuple contract. |
| `sanity_2a_50spk_v3` | 2026-11-03 | `d3d0115b` | 42 | `small_tdnn_se` | AAMSoftmax (num_classes=50, $s=30, m=0.2$) | 50-speaker subset (5,695 utts) | 410,928 (0.411M) | 21.42% (best dev ep 2) | N/A | 0.8245 (best dev norm ep 3) | Mac M3 Pro (`device = mps`, fp32) | COMPLETED [GATE PASSED] | Commit `33499df`. 3 epochs, B=32. Step-0 loss: 11.5615. Dev EER: 34.58% (ep 0) -> 24.13% -> 21.42% -> 21.58%. Unique utts: 3616 -> 4893 -> 5390. 0 NaNs. |
| `sanity_2a_full_v3` | 2026-11-03 | `70443f48` | 42 | `small_tdnn_se` | AAMSoftmax (num_classes=251, $s=30, m=0.2$) | LibriSpeech train-clean-100 (251 spk, 28,539 utts) | 410,928 (0.411M) | 10.26% (best dev ep 10) | N/A | 0.4645 (best dev norm ep 8) | Mac M3 Pro (`device = mps`, fp32) | COMPLETED [GATE PASSED] | Commit `33499df`. 10 epochs, B=64, 446 b/ep. Step-0 loss: 13.7184. Dev EER: 34.58% (ep 0) -> 10.26% (ep 10). Train-probe EER: 25.09% (ep 0) -> 3.55% (ep 10). Cum unique utts: 28,538 (99.996%). |

---

## 2. Experimental Entries

### Phase 2A Sanity Run C (v3): Full 251 Speakers (10 Epochs) [AUTHORITATIVE]
- **Run ID:** `sanity_2a_full_v3`
- **Timestamp:** 2026-11-03
- **Git Commit:** `33499df6bbb97c7882f90497a020a8933892e9da`
- **Platform:** Apple Silicon MacBook Pro 14-inch (Apple M3 Pro, 11 cores, 18 GB RAM, macOS 26.5 Build 25F71)
- **Device & Precision:** `device = mps`, `torch.float32` (PYTORCH_ENABLE_MPS_FALLBACK=1)
- **Config Hash:** `70443f48` (`configs/sanity_2a_full.yaml`)
- **Random Seed:** 42
- **Model Architecture:** `small_tdnn_se` (in_features=80, channels=128, emb_dim=192)
- **Measured Parameters (`torchinfo`):** 410,928 parameters (0.411M)
- **Loss Function:** `AAMSoftmax` (in_features=192, num_classes=251, scale=30.0, margin=0.2)
- **Dataset:** `data/manifests/train-clean-100.csv` (all 251 speakers, 28,539 total utterances)
- **Batch Size:** $B=64$ distinct speakers per batch
- **Epoch Size:** 446 batches per epoch (per ADR-016)
- **Total Steps:** 4,460 steps across 10 epochs
- **LR Schedule:** CyclicLR ($1.0\times 10^{-7} \to 1.0\times 10^{-3}$, 2-epoch cycle = 446 steps up / 446 steps down, cycle_momentum=False)
- **Active Augmentations:** speed=True (p=0.5), noise=True (p=0.3), babble=True (p=0.4), rir=False (p=0.5, explicitly OFF per instruction), specaug=True (p=0.5)
- **Data Indexing & RNG Contract:** 3-tuple `(idx, epoch, draw_id)` indexing via `SpeakerBalancedBatchSampler` and `np.random.SeedSequence([base_seed, epoch, idx, draw_id])` per ADR-020.2.
- **Initial Baseline (Step 0 & Epoch 0 Untrained):**
  - Initial Loss at Step 0: **13.7184**
  - Untrained Baseline Dev-Clean EER: **34.58%** (minDCF $P=0.05$: 0.9019)
  - Untrained Baseline Train-Probe EER: **25.09%**
- **Measured Epoch-by-Epoch Trajectory:**

| Epoch | Sampler Epoch | First Batch Hash | Unique Utts/Epoch | Cum Unique Utts | Coverage (%) | Train Loss (round) | Train Loss (unrounded) | Train Acc (%) | Train Sec | Eval Sec | Dev-Clean EER (%) | Dev minDCF ($P=0.05$) | Train-Probe EER (%) | Train vs Dev Gap | LR | Grad Norm |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 1 | `f42093f27df7f2c9` | 17,884 | 17,884 | 62.66% | 10.7060 | 10.70600980 | 10.91% | 97.1s | 67.3s | 24.03% | 0.8110 | 17.19% | +6.84% | 1.00e-3 | 13.5940 |
| 02 | 2 | `57f5c709cac7f5a0` | 17,963 | 24,603 | 86.21% | 7.4440 | 7.44402994 | 46.17% | 96.1s | 67.5s | 14.19% | 0.6610 | 8.54% | +5.65% | 1.00e-7 | 10.6906 |
| 03 | 3 | `4de4182f5b420434` | 18,002 | 27,045 | 94.77% | 6.3497 | 6.34972926 | 58.41% | 97.2s | 69.2s | 17.65% | 0.6981 | 9.95% | +7.70% | 1.00e-3 | 11.7595 |
| 04 | 4 | `c5f8e9545b02c42e` | 17,900 | 27,988 | 98.07% | 5.4071 | 5.40707274 | 65.24% | 95.1s | 70.9s | 12.35% | 0.5748 | 5.91% | +6.44% | 1.00e-7 | 10.7421 |
| 05 | 5 | `bdbb37684b4e33db` | 17,922 | 28,314 | 99.21% | 4.7886 | 4.78862348 | 70.92% | 99.5s | 73.0s | 14.13% | 0.5403 | 6.90% | +7.23% | 1.00e-3 | 10.9043 |
| 06 | 6 | `a626270b0831e094` | 17,992 | 28,451 | 99.69% | 4.3801 | 4.38007988 | 73.12% | 96.8s | 72.4s | 11.61% | 0.5155 | 4.82% | +6.79% | 1.00e-7 | 9.9607 |
| 07 | 7 | `0d94ff6fbab3ca39` | 17,929 | 28,505 | 99.88% | 3.9555 | 3.95553520 | 76.98% | 96.5s | 67.8s | 12.68% | 0.5823 | 6.32% | +6.36% | 1.00e-3 | 9.9450 |
| 08 | 8 | `425887b40fc90ddb` | 18,016 | 28,529 | 99.96% | 3.7676 | 3.76756800 | 77.91% | 96.6s | 67.2s | 10.48% | **0.4645** | 4.00% | +6.48% | 1.00e-7 | 9.2378 |
| 09 | 9 | `df4e77db7ec05d1b` | 17,939 | 28,534 | 99.98% | 3.5167 | 3.51669572 | 80.00% | 96.5s | 66.7s | 13.48% | 0.5045 | 5.40% | +8.08% | 1.00e-3 | 9.2600 |
| 10 | 10 | `5b3589c84903e6a8` | 18,062 | 28,538 | **99.996%** | 3.4160 | 3.41600386 | 80.57% | 96.0s | 67.7s | **10.26%** | 0.4810 | **3.55%** | +6.71% | 1.00e-7 | 8.7176 |

- **Re-derived Scientific Findings (v3 authoritative):**
  1. *Complete Utterance Coverage:* Cumulative unique utterances visited expands systematically across epochs ($17,884 \to 24,603 \to 27,045 \to \dots \to 28,538$), reaching 99.996% dataset coverage (28,538 out of 28,539 utterances) by epoch 10. Distinct first-batch hashes verify non-repeating batch formation across all 10 epochs.
  2. *Biometric Convergence:* Dev-clean EER progresses from $34.58\%$ untrained down to $10.26\%$ (minDCF 0.4645 at epoch 8, 0.4810 at epoch 10). Train-probe EER decreases from $25.09\%$ untrained down to $3.55\%$ at epoch 10.
  3. *LR Trough Validation Rhythm:* Validation EER and minDCF consistently reach local optima at cyclic troughs ($\text{LR} = 1.0\times 10^{-7}$, even epochs 2, 4, 6, 8, 10), while cycle peaks ($\text{LR} = 1.0\times 10^{-3}$) temporarily loosen margin separation to escape sharp local minima.
  4. *Generalization Gap:* The generalization gap between unseen evaluation (`dev-clean`) and training data (`train-probe`) remains bounded between $+5.65\%$ and $+8.08\%$ throughout training, demonstrating stable representation learning on 251 speakers without memorization collapse.
- **Artifacts:**
  - Metadata: `results/sanity_2a_full_v3/run_meta.json`
  - History JSON: `results/sanity_2a_full_v3/history.json`
  - History CSV: `results/sanity_2a_full_v3/history.csv`
  - Checkpoints: `checkpoints/sanity_2a_full_v3/epoch_{009,010}.pt`, `best_dev_eer.pt`, `final.pt`

---

### Phase 2A Sanity Run B (v3): 50-Speaker Subset (3 Epochs) [AUTHORITATIVE]
- **Run ID:** `sanity_2a_50spk_v3`
- **Timestamp:** 2026-11-03
- **Git Commit:** `33499df6bbb97c7882f90497a020a8933892e9da`
- **Platform:** Apple Silicon MacBook Pro 14-inch (Apple M3 Pro, 11 cores, 18 GB RAM, macOS 26.5 Build 25F71)
- **Device & Precision:** `device = mps`, `torch.float32` (PYTORCH_ENABLE_MPS_FALLBACK=1)
- **Config Hash:** `d3d0115b` (`configs/sanity_2a_50spk.yaml`)
- **Random Seed:** 42
- **Model Architecture:** `small_tdnn_se` (in_features=80, channels=128, emb_dim=192)
- **Measured Parameters (`torchinfo`):** 410,928 parameters (0.411M)
- **Loss Function:** `AAMSoftmax` (in_features=192, num_classes=50, scale=30.0, margin=0.2)
- **Dataset:** `data/manifests/train-50spk.csv` (first 50 speakers of train-clean-100, 5,695 total utterances)
- **Batch Size:** $B=32$ distinct speakers (*Deliberate deviation from ADR-016: 50 speakers cannot fill 64 distinct-speaker batches*)
- **Batches per Epoch:** 178 batches (one complete pass over subset utterances)
- **LR Schedule:** CyclicLR (1e-7 -> 1e-3, 2-epoch cycle = 178 steps up / 178 steps down, cycle_momentum=False)
- **Active Augmentations:** speed=True, noise=True, babble=True, rir=False (explicitly OFF), specaug=True
- **Data Indexing & RNG Contract:** 3-tuple `(idx, epoch, draw_id)` indexing via `SpeakerBalancedBatchSampler` and `np.random.SeedSequence([base_seed, epoch, idx, draw_id])` per ADR-020.2.
- **Initial Baseline (Step 0 & Epoch 0 Untrained):**
  - Initial Loss at Step 0: **11.5615**
  - Untrained Baseline Dev-Clean EER: **34.58%** (minDCF $P=0.05$: 0.9019)
- **Measured Epoch History:**
  - **Epoch 01:** Sampler Epoch = 1 | First Batch Hash = `e68e1899926439b3` | Cum Unique Utts = 3,616 | Loss = 8.7861 | Accuracy = 26.09% | LR = 1.00e-03 | Train Time = 18.1s | Eval Time = 11.2s | Dev EER = 24.13% | minDCF ($P=0.05$, norm) = 0.8381
  - **Epoch 02:** Sampler Epoch = 2 | First Batch Hash = `01d2ade5b02ef97d` | Cum Unique Utts = 4,893 | Loss = 5.6830 | Accuracy = 60.53% | LR = 1.00e-07 | Train Time = 18.6s | Eval Time = 10.4s | Dev EER = 21.42% | minDCF ($P=0.05$, norm) = 0.8252
  - **Epoch 03:** Sampler Epoch = 3 | First Batch Hash = `f5ede9df2dceb233` | Cum Unique Utts = 5,390 | Loss = 4.5137 | Accuracy = 72.00% | LR = 1.00e-03 | Train Time = 18.1s | Eval Time = 10.2s | Dev EER = 21.58% | minDCF ($P=0.05$, norm) = 0.8245
- **Diagnostic Findings & Gate Evaluation:**
  - Distinct first-batch index hashes across all 3 epochs confirm that batch formations differ per epoch under 3-tuple contract.
  - Cumulative unique utterances expand monotonically ($3,616 \to 4,893 \to 5,390$ out of 5,695, 94.6% coverage by epoch 3).
  - Training loss decreases strictly monotonically ($8.7861 \to 5.6830 \to 4.5137$).
  - Training accuracy increases steadily ($26.09\% \to 60.53\% \to 72.00\%$).
  - Best Dev EER of 21.42% achieved at epoch 2 cycle trough.
  - Zero NaNs, Infs, or gradient explosions across all 534 steps.
  - Gate Evaluation: PASSED [GATE PASSED].
- **Artifacts:**
  - History JSON: `results/sanity_2a_50spk_v3/history.json`
  - History CSV: `results/sanity_2a_50spk_v3/history.csv`
  - Checkpoints: `checkpoints/sanity_2a_50spk_v3/epoch_{002,003}.pt`, `best_dev_eer.pt`, `final.pt`

---

### Phase 2A Sanity Run A (v3): Overfit Single Fixed Batch [AUTHORITATIVE]
- **Run ID:** `sanity_2a_overfit_v3`
- **Timestamp:** 2026-11-03
- **Git Commit:** `33499df6bbb97c7882f90497a020a8933892e9da`
- **Platform:** Apple Silicon MacBook Pro 14-inch (Apple M3 Pro, 11 cores, 18 GB RAM, macOS 26.5 Build 25F71)
- **Device & Precision:** `device = mps`, `torch.float32` (PYTORCH_ENABLE_MPS_FALLBACK=1)
- **Config Hash:** `88b26f9e` (configs/sanity_2a_overfit.yaml)
- **Random Seed:** 42
- **Model Architecture:** `small_tdnn_se` (in_features=80, channels=128, emb_dim=192)
- **Measured Parameters (`torchinfo`):** 410,928 parameters (0.411M)
- **Loss Function:** `AAMSoftmax` (in_features=192, num_classes=251, scale=30.0, margin=0.2)
- **Active Augmentations:** speed=False, noise=False, babble=False, rir=False, specaug=False (all explicitly OFF)
- **Optimization:** Adam (lr=1e-3 constant, weight_decay=1e-4, grad_clip=3.0)
- **Batch Size:** 64 distinct speakers (1 fixed batch from train-clean-100)
- **Steps:** 300
- **Final Measured Metrics:**
  - Initial Loss (Step 1): 13.4322
  - Final Loss (Step 300): 0.000012
  - Loss Reduction Factor: 1,119,347.3x (threshold >= 100x)
  - Final Accuracy: 100.00% (threshold == 100%)
  - Zero NaNs, Infs, or gradient explosions across all 300 steps.
- **Artifacts:**
  - History JSON: `results/sanity_2a_overfit_v3/history.json`
  - History CSV: `results/sanity_2a_overfit_v3/history.csv`
  - Checkpoint: `checkpoints/sanity_2a_overfit_v3/overfit_final.pt`
- **Gate Evaluation:** PASSED [GATE PASSED].

---

### Phase 2A Sanity Run C (v2): Full 251 Speakers (10 Epochs) [PROVISIONAL - superseded by v3]
- **Run ID:** `sanity_2a_full_v2`
- **Timestamp:** 2026-11-03
- **Platform:** Apple Silicon MacBook Pro 14-inch (Apple M3 Pro, 11 cores, 18 GB RAM, macOS 26.5 Build 25F71)
- **Device & Precision:** `device = mps`, `torch.float32` (PYTORCH_ENABLE_MPS_FALLBACK=1)
- **Config Hash:** `70443f48` (`configs/sanity_2a_full.yaml`)
- **Random Seed:** 42
- **Model Architecture:** `small_tdnn_se` (in_features=80, channels=128, emb_dim=192)
- **Measured Parameters (`torchinfo`):** 410,928 parameters (0.411M)
- **Loss Function:** `AAMSoftmax` (in_features=192, num_classes=251, scale=30.0, margin=0.2)
- **Dataset:** `data/manifests/train-clean-100.csv` (all 251 speakers, 28,539 total utterances)
- **Batch Size:** $B=64$ distinct speakers per batch
- **Epoch Size:** 446 batches per epoch (per ADR-016)
- **Total Steps:** 4,460 steps across 10 epochs
- **LR Schedule:** CyclicLR ($1.0\times 10^{-7} \to 1.0\times 10^{-3}$, 2-epoch cycle = 446 steps up / 446 steps down, cycle_momentum=False)
- **Active Augmentations:** speed=True, noise=True, babble=True, rir=False (explicitly OFF per instruction), specaug=True
- **Initial Baseline (Step 0 & Epoch 0 Untrained):**
  - Initial Loss at Step 0: **13.6501**
  - Untrained Baseline Dev-Clean EER: **34.58%** (minDCF $P=0.05$: 0.9019)
  - Untrained Baseline Train-Probe EER: **25.09%**
- **Measured Epoch-by-Epoch Trajectory:**

| Epoch | Sampler Epoch | First Batch Hash | Cum Unique Utts | Coverage (%) | Train Loss | Train Acc (%) | Train Sec | Eval Sec | Dev-Clean EER (%) | Dev minDCF ($P=0.05$) | Train-Probe EER (%) | Train vs Dev Gap | LR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 1 | `2e8605c3f3f37376` | 17,999 | 63.07% | 10.5430 | 13.18% | 93.9s | 69.1s | 20.94% | 0.7910 | 14.97% | +5.97% | 1.00e-3 |
| 02 | 2 | `6e92bf8a8139fc20` | 24,558 | 86.05% | 7.1258 | 50.16% | 97.4s | 69.2s | 14.77% | 0.6329 | 8.03% | +6.74% | 1.00e-7 |
| 03 | 3 | `53feebfd7c778aab` | 26,979 | 94.53% | 6.1576 | 60.10% | 95.6s | 69.0s | 16.90% | 0.6997 | 8.48% | +8.42% | 1.00e-3 |
| 04 | 4 | `4d6cdaba33b6f0e2` | 27,972 | 98.01% | 5.0941 | 68.35% | 97.3s | 61.9s | 12.19% | 0.5523 | 5.71% | +6.48% | 1.00e-7 |
| 05 | 5 | `556a7c3fc4077e05` | 28,308 | 99.19% | 4.6829 | 71.83% | 97.1s | 80.9s | 15.06% | 0.6384 | 7.76% | +7.30% | 1.00e-3 |
| 06 | 6 | `890f1d4229b70f62` | 28,436 | 99.64% | 4.0899 | 76.41% | 103.0s | 64.9s | 11.84% | 0.5216 | 4.51% | +7.33% | 1.00e-7 |
| 07 | 7 | `1e727a47116b8c69` | 28,490 | 99.83% | 3.8881 | 77.61% | 93.6s | 61.0s | 12.81% | 0.5532 | 5.71% | +7.10% | 1.00e-3 |
| 08 | 8 | `6e2231c8e41e782a` | 28,518 | 99.93% | 3.5140 | 80.37% | 90.9s | 66.7s | 11.03% | **0.4548** | 3.93% | +7.10% | 1.00e-7 |
| 09 | 9 | `7a5a05bc078558df` | 28,531 | 99.97% | 3.4372 | 80.69% | 93.3s | 69.6s | 12.81% | 0.5371 | 5.06% | +7.75% | 1.00e-3 |
| 10 | 10 | `82f6b76a9189691a` | 28,536 | **99.99%** | 3.1730 | 82.85% | 95.8s | 66.6s | **10.77%** | 0.4577 | **3.59%** | +7.18% | 1.00e-7 |

- **Re-derived Scientific Findings (v2 only):**
  1. *Complete Utterance Coverage:* Cumulative unique utterances visited expands systematically across epochs ($17,999 \to 24,558 \to 26,979 \to \dots \to 28,536$), reaching 99.99% dataset coverage by epoch 10. Distinct first-batch hashes verify non-repeating batch formation across all 10 epochs.
  2. *Biometric Convergence:* Dev-clean EER progresses from $34.58\%$ untrained down to $10.77\%$ (minDCF 0.4548 at epoch 8, 0.4577 at epoch 10). Train-probe EER decreases from $25.09\%$ untrained down to $3.59\%$ at epoch 10.
  3. *LR Trough Validation Rhythm:* Validation EER and minDCF consistently reach local optima at cyclic troughs ($\text{LR} = 1.0\times 10^{-7}$, even epochs 2, 4, 6, 8, 10), while cycle peaks ($\text{LR} = 1.0\times 10^{-3}$) temporarily loosen margin separation to escape sharp local minima.
  4. *Generalization Gap:* The generalization gap between unseen evaluation (`dev-clean`) and training data (`train-probe`) remains bounded between $+5.97\%$ and $+8.42\%$ throughout training, demonstrating stable representation learning on 251 speakers without memorization collapse.
- **Artifacts:**
  - Metadata: `results/sanity_2a_full_v2/run_meta.json`
  - History JSON: `results/sanity_2a_full_v2/history.json`
  - History CSV: `results/sanity_2a_full_v2/history.csv`
  - Checkpoints: `checkpoints/sanity_2a_full_v2/epoch_{009,010}.pt`, `best_dev_eer.pt`, `final.pt`

---

### Phase 2A Sanity Run C: Full 251 Speakers (10 Epochs) [SUPERSEDED - set_epoch bug]
> [!WARNING]
> **SUPERSEDED by `sanity_2a_full_v2`.** In this historical run, `train_loader.sampler.set_epoch` was called instead of `train_loader.batch_sampler.set_epoch`. Because PyTorch assigns a default `SequentialSampler` to `.sampler` when `batch_sampler` is provided, `set_epoch` was never invoked on `SpeakerBalancedBatchSampler`. Furthermore, under `persistent_workers=True`, worker processes maintained `epoch = 0`. As a result, the exact same batch sequence was replayed every epoch, and only 18,045 out of 28,539 unique utterances (63.23%) were ever drawn across the 10 epochs (36.77% of training data was never seen). All findings from this run are superseded by `sanity_2a_full_v2`.

- **Run ID:** `sanity_2a_full`
- **Timestamp:** 2026-11-03
- **Platform:** Apple Silicon MacBook Pro 14-inch (Apple M3 Pro, 11 cores, 18 GB RAM, macOS 26.5 Build 25F71)
- **Device & Precision:** `device = mps`, `torch.float32` (PYTORCH_ENABLE_MPS_FALLBACK=1)
- **Config Hash:** `70443f48` (`configs/sanity_2a_full.yaml`)
- **Random Seed:** 42
- **Model Architecture:** `small_tdnn_se` (in_features=80, channels=128, emb_dim=192)
- **Measured Parameters (`torchinfo`):** 410,928 parameters (0.411M)
- **Loss Function:** `AAMSoftmax` (in_features=192, num_classes=251, scale=30.0, margin=0.2)
- **Dataset:** `data/manifests/train-clean-100.csv` (all 251 speakers, 28,539 total utterances)
- **Batch Size:** $B=64$ distinct speakers per batch
- **Epoch Size:** 446 batches per epoch (per ADR-016)
- **Total Steps:** 4,460 steps across 10 epochs
- **LR Schedule:** CyclicLR ($1.0\times 10^{-7} \to 1.0\times 10^{-3}$, 2-epoch cycle = 446 steps up / 446 steps down, cycle_momentum=False)
- **Active Augmentations:** speed=True, noise=True, babble=True, rir=False (explicitly OFF per instruction), specaug=True
- **Measured Epoch-by-Epoch Trajectory:**

| Epoch | Train Loss | Train Acc (%) | Train Sec | Eval Sec | Dev-Clean EER (%) | Dev minDCF ($P=0.05$) | Train-Probe EER (%) | Train vs Dev Gap (Dev - TrainProbe) | LR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 10.5741 | 12.57% | 95.1s | 173.9s | 23.84% | 0.8168 | 16.44% | +7.40% | 1.00e-3 |
| 02 | 7.0195 | 51.24% | 98.2s | 65.0s | 14.52% | 0.6184 | 8.34% | +6.18% | 1.00e-7 |
| 03 | 5.6459 | 65.79% | 92.5s | 61.3s | 16.68% | 0.6797 | 8.54% | +8.14% | 1.00e-3 |
| 04 | 4.6334 | 72.93% | 93.2s | 63.6s | 12.03% | 0.5335 | 6.08% | +5.95% | 1.00e-7 |
| 05 | 3.8688 | 79.31% | 94.9s | 64.0s | 13.90% | 0.6352 | 6.90% | +7.00% | 1.00e-3 |
| 06 | 3.5563 | 81.00% | 96.3s | 71.1s | 11.16% | 0.4819 | 5.02% | +6.14% | 1.00e-7 |
| 07 | 2.9805 | 85.65% | 97.2s | 67.6s | 13.71% | 0.6161 | 6.08% | +7.63% | 1.00e-3 |
| 08 | 2.9404 | 85.51% | 96.4s | 66.2s | 11.45% | 0.4658 | 4.24% | +7.21% | 1.00e-7 |
| 09 | 2.4845 | 88.79% | 95.6s | 67.0s | 14.61% | 0.5003 | 6.22% | +8.39% | 1.00e-3 |
| 10 | 2.5086 | 88.50% | 96.0s | 64.6s | **10.71%** | **0.4539** | **3.90%** | +6.81% | 1.00e-7 |

- **Diagnostic Findings & Methodological Interpretation:**
  1. *Optimization Dynamics:* The loss drops by an order of magnitude ($10.5741 \to 2.5086$) while training accuracy increases from $12.57\%$ to $88.50\%$. Zero NaNs, Infs, or gradient explosions occurred across all 4,460 steps.
  2. *Cyclic Learning Rate Behavior:* The 2-epoch triangular cycle exhibits distinct valley vs. peak dynamics:
     - At cycle troughs ($\text{LR} = 1.0\times 10^{-7}$, even epochs 2, 4, 6, 8, 10), validation metrics systematically reach their best values (Dev EER progressing $14.52\% \to 12.03\% \to 11.16\% \to 11.45\% \to 10.71\%$).
     - At cycle peaks ($\text{LR} = 1.0\times 10^{-3}$, odd epochs 1, 3, 5, 7, 9), validation metrics temporarily loosen (Dev EER 13.7%–16.7%), which enables escape from sharp local minima into flatter, more generalizable basins.
  3. *Generalization Gap Analysis (Train-Probe vs. Dev-Clean):*
     - Training-data EER (`train-probe`: 5,852 cross-chapter trials on training speakers) progresses from $16.44\% \to 3.90\%$.
     - Unseen validation EER (`dev-clean`: 5,800 trials on 31 unseen speakers) progresses from $23.84\% \to 10.71\%$.
     - The train-vs-dev generalization gap remains strictly bounded between $+5.95\%$ and $+8.39\%$ throughout training. Even without RIR reverberation and on only 251 training speakers, there is no evidence of catastrophic memorization or representation collapse.
  4. *Runtime & Throughput:*
     - Training time per epoch was remarkably steady: $95.0\pm 1.8\text{ s}$ per epoch ($213\text{ ms/batch}$ on MPS with on-the-fly multi-augmentation decoding).
     - Evaluation time: Epoch 1 disk load took $173.9\text{ s}$; once cached in macOS RAM, dual evaluation of 2,145 dev + 3,000 train-probe utterances took only $65.4\pm 3.1\text{ s}$.
  5. *Gate Evaluation:* PASSED [GATE PASSED]. All technical gates satisfied: passing unit tests, zero NaNs, bitwise checkpoint resume, and expected biometric convergence on dev-clean.
- **Artifacts:**
  - History JSON: `results/sanity_2a_full/history.json`
  - History CSV: `results/sanity_2a_full/history.csv`
  - Checkpoints: `checkpoints/sanity_2a_full/epoch_{001..010}.pt`, `best_dev_eer.pt`

---

### Phase 2A Sanity Run B (v2): 50-Speaker Subset (3 Epochs) [PROVISIONAL - superseded by v3]
- **Run ID:** `sanity_2a_50spk_v2`
- **Timestamp:** 2026-11-03
- **Platform:** Apple Silicon MacBook Pro 14-inch (Apple M3 Pro, 11 cores, 18 GB RAM, macOS 26.5 Build 25F71)
- **Device & Precision:** `device = mps`, `torch.float32` (PYTORCH_ENABLE_MPS_FALLBACK=1)
- **Config Hash:** `d3d0115b` (`configs/sanity_2a_50spk.yaml`)
- **Random Seed:** 42
- **Model Architecture:** `small_tdnn_se` (in_features=80, channels=128, emb_dim=192)
- **Measured Parameters (`torchinfo`):** 410,928 parameters (0.411M)
- **Loss Function:** `AAMSoftmax` (in_features=192, num_classes=50, scale=30.0, margin=0.2)
- **Dataset:** `data/manifests/train-50spk.csv` (first 50 speakers of train-clean-100, 5,695 total utterances)
- **Batch Size:** $B=32$ distinct speakers (*Deliberate deviation from ADR-016: 50 speakers cannot fill 64 distinct-speaker batches*)
- **Batches per Epoch:** 178 batches (one complete pass over subset utterances)
- **LR Schedule:** CyclicLR (1e-7 -> 1e-3, 2-epoch cycle = 178 steps up / 178 steps down, cycle_momentum=False)
- **Active Augmentations:** speed=True, noise=True, babble=True, rir=False (explicitly OFF), specaug=True
- **Initial Baseline (Step 0 & Epoch 0 Untrained):**
  - Initial Loss at Step 0: **11.5149**
  - Untrained Baseline Dev-Clean EER: **34.58%** (minDCF $P=0.05$: 0.9019)
- **Measured Epoch History:**
  - **Epoch 01:** Sampler Epoch = 1 | First Batch Hash = `d46f94d9a5f05e49` | Cum Unique Utts = 3,609 | Loss = 8.4377 | Accuracy = 30.30% | LR = 1.00e-03 | Train Time = 18.5s | Eval Time = 11.8s | Dev EER = 25.45% | minDCF ($P=0.05$, norm) = 0.8471
  - **Epoch 02:** Sampler Epoch = 2 | First Batch Hash = `bf25f4b4587de81a` | Cum Unique Utts = 4,921 | Loss = 5.2722 | Accuracy = 64.45% | LR = 1.00e-07 | Train Time = 19.7s | Eval Time = 11.4s | Dev EER = 20.39% | minDCF ($P=0.05$, norm) = 0.7813
  - **Epoch 03:** Sampler Epoch = 3 | First Batch Hash = `7fafad244dee0d04` | Cum Unique Utts = 5,391 | Loss = 4.3030 | Accuracy = 73.61% | LR = 1.00e-03 | Train Time = 18.4s | Eval Time = 10.0s | Dev EER = 22.74% | minDCF ($P=0.05$, norm) = 0.7642
- **Diagnostic Findings & Gate Evaluation:**
  - Distinct first-batch index hashes across all 3 epochs confirm that batch formations differ per epoch.
  - Cumulative unique utterances expand monotonically ($3,609 \to 4,921 \to 5,391$ out of 5,695, 94.7% coverage by epoch 3).
  - Training loss decreases strictly monotonically ($8.4377 \to 5.2722 \to 4.3030$).
  - Training accuracy increases steadily ($30.30\% \to 64.45\% \to 73.61\%$).
  - Best Dev EER of 20.39% achieved at epoch 2 cycle trough.
  - Zero NaNs, Infs, or gradient explosions across all 534 steps.
  - Gate Evaluation: PASSED [GATE PASSED].
- **Artifacts:**
  - History JSON: `results/sanity_2a_50spk_v2/history.json`
  - History CSV: `results/sanity_2a_50spk_v2/history.csv`
  - Checkpoints: `checkpoints/sanity_2a_50spk_v2/epoch_{002,003}.pt`, `best_dev_eer.pt`, `final.pt`

---

### Phase 2A Sanity Run B: 50-Speaker Subset (3 Epochs) [SUPERSEDED - set_epoch bug]
> [!WARNING]
> **SUPERSEDED by `sanity_2a_50spk_v2`.** In this historical run, `train_loader.sampler.set_epoch` was checked instead of `train_loader.batch_sampler.set_epoch`, meaning `SpeakerBalancedBatchSampler` never received epoch updates and replayed an identical batch sequence every epoch. All conclusions drawn from this run are superseded by `sanity_2a_50spk_v2`.

- **Run ID:** `sanity_2a_50spk`
- **Timestamp:** 2026-11-03
- **Platform:** Apple Silicon MacBook Pro 14-inch (Apple M3 Pro, 11 cores, 18 GB RAM, macOS 26.5 Build 25F71)
- **Device & Precision:** `device = mps`, `torch.float32` (PYTORCH_ENABLE_MPS_FALLBACK=1)
- **Config Hash:** `d3d0115b` (configs/sanity_2a_50spk.yaml)
- **Random Seed:** 42
- **Model Architecture:** `small_tdnn_se` (in_features=80, channels=128, emb_dim=192)
- **Measured Parameters (`torchinfo`):** 410,928 parameters (0.411M)
- **Loss Function:** `AAMSoftmax` (in_features=192, num_classes=50, scale=30.0, margin=0.2)
- **Dataset:** `data/manifests/train-50spk.csv` (first 50 speakers of train-clean-100, 5,695 total utterances)
- **Batch Size:** $B=32$ distinct speakers (*Deliberate deviation from ADR-016: 50 speakers cannot fill 64 distinct-speaker batches*)
- **Batches per Epoch:** 178 batches (one complete pass over subset utterances)
- **LR Schedule:** CyclicLR (1e-7 -> 1e-3, 2-epoch cycle = 178 steps up / 178 steps down, cycle_momentum=False)
- **Active Augmentations:** speed=True, noise=True, babble=True, rir=False (explicitly OFF), specaug=True
- **Measured Epoch History:**
  - **Epoch 01:** Loss = 8.7093 | Accuracy = 26.04% | LR = 1.00e-03 | Train Time = 18.8s | Eval Time = 54.0s | Dev EER = 26.16% | minDCF ($P=0.05$, norm) = 0.8858
  - **Epoch 02:** Loss = 5.1842 | Accuracy = 65.19% | LR = 1.00e-07 | Train Time = 17.6s | Eval Time = 9.9s | Dev EER = 20.58% | minDCF ($P=0.05$, norm) = 0.8055
  - **Epoch 03:** Loss = 3.6097 | Accuracy = 78.12% | LR = 1.00e-03 | Train Time = 17.8s | Eval Time = 9.8s | Dev EER = 23.35% | minDCF ($P=0.05$, norm) = 0.8094
- **Diagnostic Findings & Gate Evaluation:**
  - Training loss decreases strictly monotonically ($8.7093 \to 5.1842 \to 3.6097$).
  - Training accuracy increases steadily ($26.04\% \to 65.19\% \to 78.12\%$).
  - Zero NaNs, Infs, or gradient explosions across all 534 optimization steps.
  - Full-utterance extraction on unseen `dev-clean` speakers evaluates cleanly, reaching 20.58% EER at epoch 2.
  - Evaluation caching functions effectively: evaluation time drops from 54.0s (first epoch disk load) to 9.8s in subsequent epochs.
  - Gate Evaluation: PASSED [GATE PASSED].
- **Artifacts:**
  - History JSON: `results/sanity_2a_50spk/history.json`
  - History CSV: `results/sanity_2a_50spk/history.csv`
  - Checkpoints: `checkpoints/sanity_2a_50spk/epoch_{001,002,003}.pt`, `best_dev_eer.pt`

---

### Phase 2A Sanity Run A: Overfit Single Fixed Batch
- **Run ID:** `sanity_2a_overfit`
- **Timestamp:** 2026-11-03
- **Platform:** Apple Silicon MacBook Pro 14-inch (Apple M3 Pro, 11 cores, 18 GB RAM, macOS 26.5 Build 25F71)
- **Device & Precision:** `device = mps`, `torch.float32` (PYTORCH_ENABLE_MPS_FALLBACK=1)
- **Config Hash:** `88b26f9e` (configs/sanity_2a_overfit.yaml)
- **Random Seed:** 42
- **Model Architecture:** `small_tdnn_se` (in_features=80, channels=128, emb_dim=192)
- **Measured Parameters (`torchinfo`):** 410,928 parameters (0.411M)
- **Loss Function:** `AAMSoftmax` (in_features=192, num_classes=251, scale=30.0, margin=0.2)
- **Active Augmentations:** speed=False, noise=False, babble=False, rir=False, specaug=False (all explicitly OFF)
- **Optimization:** Adam (lr=1e-3 constant, weight_decay=1e-4, grad_clip=3.0)
- **Batch Size:** 64 distinct speakers (1 fixed batch from train-clean-100)
- **Steps:** 300
- **Measured Loss Trajectory:**
  - Step 001: Loss = 13.3750 | Accuracy = 0.00% | GradNorm = 26.016 | Elapsed = 1.5s
  - Step 025: Loss = 0.000412 | Accuracy = 100.00% | GradNorm = 0.011 | Elapsed = 2.3s
  - Step 050: Loss = 0.000096 | Accuracy = 100.00% | GradNorm = 0.002 | Elapsed = 3.1s
  - Step 075: Loss = 0.000045 | Accuracy = 100.00% | GradNorm = 0.001 | Elapsed = 4.0s
  - Step 100: Loss = 0.000028 | Accuracy = 100.00% | GradNorm = 0.000 | Elapsed = 4.8s
  - Step 150: Loss = 0.000017 | Accuracy = 100.00% | GradNorm = 0.000 | Elapsed = 6.5s
  - Step 200: Loss = 0.000013 | Accuracy = 100.00% | GradNorm = 0.000 | Elapsed = 8.2s
  - Step 250: Loss = 0.000012 | Accuracy = 100.00% | GradNorm = 0.000 | Elapsed = 10.0s
  - Step 300: Loss = 0.000011 | Accuracy = 100.00% | GradNorm = 0.000 | Elapsed = 11.7s
- **Final Measured Metrics:**
  - Initial Loss: 13.3750
  - Final Loss: 0.000011
  - Loss Reduction Factor: 1,207,242.4x (threshold >= 100x)
  - Final Accuracy: 100.00% (threshold == 100%)
  - Zero NaNs, Infs, or gradient explosions across all 300 steps.
- **Artifacts:**
  - History JSON: `results/sanity_2a_overfit/history.json`
  - History CSV: `results/sanity_2a_overfit/history.csv`
  - Checkpoint: `checkpoints/sanity_2a_overfit/overfit_final.pt`
- **Gate Evaluation:** PASSED [GATE PASSED]. Confirms end-to-end model forward/backward, AAMSoftmax margin gradient dynamics, optimizer step, and MPS fp32 numerical stability.

---

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

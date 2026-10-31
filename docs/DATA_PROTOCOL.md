# DATA_PROTOCOL.md — Data Engineering & Partitioning Protocol

**Primary Dataset:** LibriSpeech ASR Corpus (OpenSLR 12)  
**Target Hardware:** Apple Mac M3 Pro (Unified Memory, fast internal NVMe SSD)

---

## 1. Corpus Partitions & Split Specifications

LibriSpeech is selected for its native 16 kHz sampling rate (matching the paper's 80-Mel / 25 ms / 10 ms front-end `[Paper]`) and its preserved natural pauses `[Engineering Choice]`.

> [!NOTE]
> Speaker, gender, hour and file counts in the table are from memory/the PRD and are `[External-UNVERIFIED]`. The manifest builder must compute them from the data and log them. They must replace these values.

| Partition Name | Purpose | Number of Speakers | Audio Duration | Raw Format | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`train-clean-100`** | Supervised & SSL Training | 251 (126 M / 125 F) | ~100.6 hours (28,539 FLACs) | 16 kHz, 16-bit FLAC | Full training corpus. |
| **`fast-dev-subset`** | Fast-Cycle Smoke Testing | 50 (25 M / 25 F) | ~20 hours (~5,600 FLACs) | 16 kHz, 16-bit FLAC | Formal subset for rapid CI and debug cycles `[Engineering Choice]`. |
| **`dev-clean`** | Validation & Hyperparameter Tuning | 40 (20 M / 20 F) | ~5.4 hours (2,703 FLACs) | 16 kHz, 16-bit FLAC | Disjoint speakers; model checkpointing. |
| **`test-clean`** | Primary Clean Verification Benchmark | 40 (20 M / 20 F) | ~5.4 hours (2,620 FLACs) | 16 kHz, 16-bit FLAC | Standard evaluation test set. |
| **`test-other`** | Acoustic Robustness Benchmark | 33 (17 M / 16 F) | ~5.1 hours (2,939 FLACs) | 16 kHz, 16-bit FLAC | Higher acoustic noise, variable recording conditions. |

### 1.1 Speaker-Disjointness Assertion
All splits must have strictly zero overlap in speaker IDs:
$$\text{Speakers}(\text{train}) \cap \text{Speakers}(\text{dev}) \cap \text{Speakers}(\text{test-clean}) \cap \text{Speakers}(\text{test-other}) = \emptyset$$
This constraint is asserted in unit test `tests/test_data_integrity.py`.

---

## 2. Manifest Schema

Data preparation produces JSON Lines manifests (`data/manifests/*.jsonl`) with the following standardized fields:

```json
{
  "utt_id": "19-198-0001",
  "speaker_id": "19",
  "gender": "M",
  "chapter_id": "198",
  "audio_path": "data/raw/LibriSpeech/train-clean-100/19/198/19-198-0001.flac",
  "duration_sec": 14.23,
  "num_frames": 1421,
  "sampling_rate": 16000
}
```

---

## 3. Verification Trial Generation Protocol

Standard verification evaluation measures cosine similarity between pairs of utterances. To prevent models from exploiting superficial acoustic cues, trial generation follows strict constraints:

### 3.1 Strict Cross-Chapter Target Trials
- In LibriSpeech, all utterances within a chapter share identical room impulse responses, background room tone, and microphone placement.
- **Strict Rule:** Target pairs ($y_1 = y_2$) must ALWAYS come from different chapters ($\text{chapter}_1 \neq \text{chapter}_2$).
- **Single-Chapter Speaker Policy:** Speakers appearing in only a single chapter are excluded from genuine target trials, so 100% of target trials are cross-chapter `[Engineering Choice]`. The builder must **log the number and fraction of speakers discarded** per test set (and whether any impostor trials still use them).

### 3.2 Gender-Matched Impostor Trials
- Impostor pairs ($y_1 \neq y_2$) are strictly sampled from different speakers of the **same gender** ($\text{gender}_1 = \text{gender}_2$). This eliminates trivial pitch-based gender discrimination and forces biometric speaker identity verification.
- Utterances are sampled uniformly across speakers to avoid over-representing prolific speakers.

### 3.3 Trial List Volumes
- Trial counts: TBD: determined by the builder and logged; no target count is assumed.
- `test-other` is an acoustic-robustness set. It is not an analogue of VoxCeleb1-H (hard impostor pairs) `[Engineering Choice]`.
- Manifest format:
  ```text
  1 data/raw/LibriSpeech/test-clean/61/70968/61-70968-0000.flac data/raw/LibriSpeech/test-clean/61/70970/61-70970-0001.flac
  0 data/raw/LibriSpeech/test-clean/61/70968/61-70968-0000.flac data/raw/LibriSpeech/test-clean/121/121726/121-121726-0003.flac
  ```

---

## 4. Hardware-Sized Caching & Disk Strategy (Mac M3 Pro)

Based on empirical Phase 1A measurements logged in [docs/BENCHMARKS.md](BENCHMARKS.md):

1. **Storage Footprint & Safety Margin:**
   - Available free disk space on the host Mac M3 Pro is **~27.5–27.9 GB** `[Measured]`.
   - **Tarball Cleanup Policy:** Always delete archive tarballs (e.g. `*.tar.gz`, `*.zip`) immediately after verified extraction to prevent redundant storage consumption `[Engineering Choice]`.
   - **Minimum Free Disk Buffer:** Always preserve at least **$\ge 8\text{ GB}$** of free disk space across all data preparation and training stages to accommodate macOS unified memory virtual paging/swap `[Engineering Choice]`. Downloads over 1 GB require explicit user confirmation prior to execution.
2. **On-the-Fly Audio Decoding (No int16 Memmap Cache):**
   - **Decision:** Do **NOT** build an int16 memmap cache on disk `[Engineering Choice]`.
   - *Rationale:* DataLoader throughput measurements in `docs/BENCHMARKS.md` confirm that on-the-fly FLAC decoding (`soundfile`) and CPU 80-Mel extraction take **$26.36\text{ ms/batch}$** at `num_workers = 4`, whereas forward+backward step times are **$\ge 118.61\text{ ms/step}$** (up to ~415 ms with full SKA branches). Because data loading is 4.5x to 15x faster than compute, asynchronous background pre-fetching completely hides I/O latency without needing an ~11.6 GB int16 memmap cache.
3. **DataLoader Rules on macOS** `[Engineering Choice]`:
   - `num_workers = 4`, `persistent_workers = True`, `pin_memory = False`.
   - Segments are 180 frames ($29,040\text{ samples}$ @ 16 kHz) cropped randomly per training step `[Paper]`.
4. **Google Colab Fallback** `[Engineering Choice]`:
   - Training is performed locally on Mac M3 Pro. Colab T4 is maintained solely as a secondary fallback (`notebooks/colab_runner.ipynb`); if activated, data is copied as a single archive into `/content` to avoid per-file Google Drive latency.

---

## 5. Augmentation Pipeline (Low-Footprint, High-Fidelity)

Instead of downloading the 11 GB MUSAN archive, an authentic yet compact augmentation pipeline is deployed:
1. **RIR Reverberation:**
   - Download official `RIRS_NOISES.zip` (1.3 GB, OpenSLR 28).
   - Sample simulated small/medium room impulse responses and apply on-the-fly convolution via `torchaudio.functional.fftconvolve` with probability $p=0.5$.
2. **Synthetic Noise Synthesizer (Zero-Download):**
   - **In-Batch Multi-Speaker Babble:** Randomly sum 3 to 5 audio slices from other speakers in the training batch at SNR 10–20 dB with probability $p=0.4$.
   - **Gaussian White & Pink Noise:** Generate on-the-fly additive colored noise at SNR 5–15 dB with probability $p=0.3$.
3. **Speed Perturbation:**
   - Random speed resample factors $\in [0.95, 1.05]$ using `torchaudio.transforms.Resample` with probability $p=0.5$.
4. **SpecAugment:**
   - Applied directly to computed 80-dim log-Mel filterbanks: 2 frequency masks (max width 10) and 2 time masks (max width 15) with probability $p=0.5$.

---

## 6. Tier 3 Self-Supervised Leakage Checks

To prevent supervised label leakage in Tier 3 self-supervised experiments:
1. **Feature Source:** Pseudo-labels come from clustering embeddings of a frozen self-supervised model (`WavLM-Base+`) trained without speaker labels.
   - Caveat (`[External-UNVERIFIED]`): WavLM-Base+ pretraining data likely includes Libri-Light/LibriVox audio that may overlap LibriSpeech. This is label-free but a contamination risk. It is recorded in OPEN_QUESTIONS.
   - The cluster count K=251 equals the true speaker count. The paper does the same (5,994 = VoxCeleb2 speaker count `[Paper]`). We also report other K values as a sensitivity check.
2. **Leakage Verification Unit Test:** `tests/test_ssl_leakage.py` asserts that no speaker IDs, classifier checkpoints, or Tier 0/1 weights were used to generate pseudo-labels.
3. **Purity & NMI Logging:** Every pseudo-label manifest is evaluated against true ground-truth speaker IDs to log cluster purity:
   $$\text{Purity} = \frac{1}{N} \sum_{k} \max_j |c_k \cap s_j|$$
   and Normalized Mutual Information ($\text{NMI}$).

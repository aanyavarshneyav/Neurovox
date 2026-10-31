# RESULTS_LOG.md — Experimental Run Ledger

**Integrity Rule:** No number may be cited as an experimental result unless it is logged in this ledger with an active `run_id`, configuration hash, seed, and measured metrics. Anticipated or literature numbers are strictly labeled `HYPOTHESIS`.

---

## 1. Experimental Ledger Template

| Run ID | Timestamp | Config Hash | Seed | Model Architecture | Loss Function | Dataset / Condition | Measured Params (`torchinfo`) | EER (%) | minDCF ($P=0.01$) | minDCF ($P=0.05$) | Device / Hardware | Run Status | Notes & Hypotheses |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *Template* | *YYYY-MM-DD* | *sha256* | *42* | *scaled_ska_hnn* | *iaam_softmax* | *LibriSpeech test-clean* | *TBD: measure* | *TBD* | *TBD* | *TBD* | *Mac M3 Pro (mps)* | *PENDING* | *Initial run template* |

---

## 2. Experimental Entries

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

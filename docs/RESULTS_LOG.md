# RESULTS_LOG.md — Experimental Run Ledger

**Integrity Rule:** No number may be cited as an experimental result unless it is logged in this ledger with an active `run_id`, configuration hash, seed, and measured metrics. Anticipated or literature numbers are strictly labeled `HYPOTHESIS`.

---

## 1. Experimental Ledger Template

| Run ID | Timestamp | Config Hash | Seed | Model Architecture | Loss Function | Dataset / Condition | Measured Params (`torchinfo`) | EER (%) | minDCF ($P=0.01$) | minDCF ($P=0.05$) | Device / Hardware | Run Status | Notes & Hypotheses |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *Template* | *YYYY-MM-DD* | *sha256* | *42* | *scaled_ska_hnn* | *iaam_softmax* | *LibriSpeech test-clean* | *TBD: measure* | *TBD* | *TBD* | *TBD* | *Mac M3 Pro (mps)* | *PENDING* | *Initial run template* |

---

## 2. Experimental Entries

*(Entries will be appended chronologically as runs complete via `.agents/workflows/new-experiment.md`.)*

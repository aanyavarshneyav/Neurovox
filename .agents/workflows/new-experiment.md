# Workflow: /new-experiment

**Purpose:** Configure, assign a standardized run ID to, execute, and atomically log a new training or evaluation experiment.

---

## Step-by-Step Procedure

1. **Create Experiment Configuration File:**
   - Create a new YAML configuration under `configs/`:
     ```bash
     configs/<tier>_<model>_<loss>_<dataset>_<condition>_s<seed>.yaml
     ```
   - Inherit defaults from base configuration (`configs/base.yaml`).
   - Specify:
     - `model`: architecture variant (e.g. `skahnn`, `ecapa`), channel widths ($C, C_1$), ablation switches.
     - `loss`: `iaam_softmax` or `aam_softmax`, scale $s=30$, margin $m=0.2$, $\tau=0.423$, $ESD=5$, $ESC=5$.
     - `training`: batch size (64), initial lr ($10^{-7}$), max lr ($10^{-3}$), epochs (30), seed (42/43/44).
     - `data`: manifests for train, dev, and test.

2. **Compute Configuration Hash:**
   ```bash
   shasum -a 256 configs/<config_name>.yaml | cut -c 1-8
   ```

3. **Initialize Results Log Entry:**
   - Open `docs/RESULTS_LOG.md` and append a row with status `RUNNING`.

4. **Launch Training:**
   ```bash
   python scripts/train.py --config configs/<config_name>.yaml
   ```

5. **Execute Evaluation:**
   - Run verification evaluation across `test-clean` and `test-other`:
     ```bash
     python scripts/evaluate.py \
       --checkpoint checkpoints/<run_id>/best_eer.pt \
       --trials data/manifests/trials_test_clean.txt data/manifests/trials_test_other.txt \
       --out_dir outputs/<run_id>
     ```

6. **Log Measured Metrics:**
   - Compute EER (%), minDCF ($P=0.01$), and minDCF ($P=0.05$).
   - Run paired bootstrap CI (1,000 resamples).
   - Update `docs/RESULTS_LOG.md` with final measured values and change status to `COMPLETED`.

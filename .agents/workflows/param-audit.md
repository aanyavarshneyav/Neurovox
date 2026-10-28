# Workflow: /param-audit

**Purpose:** Instantiate all network architectures and layers, execute a layer-by-layer `torchinfo` parameter inspection, and update the parameter count tables in `docs/ARCHITECTURE.md` and `docs/PROJECT_SPEC.md` with verified, measured numbers.

---

## Prerequisites
- Virtual environment active with `torch` and `torchinfo` installed.
- Models defined in `src/models/`.

---

## Step-by-Step Procedure

1. **Execute Parameter Audit Script:**
   ```bash
   python scripts/audit_parameters.py --device cpu --input_shape 1 1 80 180
   ```

2. **Inspect Layer-by-Layer Output:**
   Verify `torchinfo.summary()` reports for:
   - Preceding Conv2d Block
   - SKA ResBlock 1 & 2 (with fwSKA, cwSKA, and SE)
   - Succeeding Conv2d Block
   - TDNN-LSTM Hybrid Module
   - Fully-TDNN Module
   - Multi-Level Attentive Statistics Pooling (MLASP)
   - Fully Connected Projection Layers (FC1 and FC2)
   - Full End-to-End SKA-HNN Model (Scaled: $C=32, C_1=256$)
   - Ablation Variants (S2, S3, S4, S5, S6)
   - Baseline Models (Parameter-matched ECAPA-TDNN and Standard ECAPA-TDNN)

3. **Record Total Model Parameters:**
   - Record exact trainable/non-trainable parameters and MACs from `torchinfo`. Do not compare to any expected value.
   - Use the measured scaled SKA-HNN count to choose the parameter-matched ECAPA-TDNN width.

4. **Update Documentation Tables:**
   - Open `docs/ARCHITECTURE.md` and replace `"TBD: measure with torchinfo"` in Table 2 with the measured numbers.
   - Open `docs/PROJECT_SPEC.md` and update the Scaled Project column in Table 1.3.
   - Append audit log to `docs/RESULTS_LOG.md` under Run ID `audit_torchinfo_<date>`.

5. **Self-Check:**
   - Confirm no hand-calculated parameter values were inserted.
   - Check git diff to ensure only parameter count numbers were updated.

# Resolution of Phase 1B Review Findings

This document records the exact resolution, code modifications, and verifying tests for each finding identified during the Phase 1B methodological audit (`docs/reviews/1B_review.md`).

---

## 1. Priority Findings Resolution Table

| Finding ID | Description | Resolution & Implementation Details | Modified Files & Lines | Verifying Test(s) |
| :--- | :--- | :--- | :--- | :--- |
| **H1** | Inverted score/direction check did nothing (`pass`). | Added active score-direction validation: compares `mean(target)` vs `mean(impostor)`. If `mean_pos < mean_neg`, emits loud `logger.warning(...)` by default, and raises `ValueError("Inverted score distribution...")` when `strict=True`. | `src/eval/metrics.py`<br>(lines 80–92, `_validate_inputs`, `compute_eer`, `compute_mindcf`) | `tests/test_prefix_review_failures.py::test_h1_direction_check_fails_loudly_on_inverted_scores`<br>`tests/test_metrics.py::test_label_direction_inversion` |
| **H2** | `minDCF` unnormalized by default (`normalize=False`); ambiguous logging. | Introduced `MinDCFResult(dict)` returning both normalized (`min_dcf`) and unnormalized (`min_dcf_raw`) metrics alongside `threshold` and `p_target`. Set `normalize=True` as default. Primary prior established as $P_{\text{target}}=0.05$; $0.01$ designated indicative. | `src/eval/metrics.py`<br>(lines 19–49, 140–210) | `tests/test_prefix_review_failures.py::test_h2_mindcf_is_normalized_by_default_with_named_keys`<br>`tests/test_metrics.py::test_perfect_separation` |
| **H3** | Clustered bootstrap too narrow (single cluster per trial; missed 2-speaker impostor dependence). | Implemented full SPEAKER-LEVEL clustered bootstrap with per-trial weights: target trials receive weight $w_i = \text{count}(s_a)$; impostor trials receive weight $w_i = \text{count}(s_a) \times \text{count}(s_b)$. Implemented $O(N \log N)$ weighted EER and minDCF. Evaluates competing models under identical speaker resample weights. Reports point estimates, bootstrap means, 95% CIs, and paired differences. Documented limitations in `docs/EXPERIMENT_PLAN.md`. | `src/eval/metrics.py`<br>(lines 350–550, `speaker_bootstrap_ci`, `speaker_bootstrap_two_sample_difference`) | `tests/test_metrics.py::test_speaker_bootstrap_ci`<br>`tests/test_metrics.py::test_speaker_bootstrap_two_sample_difference` |
| **H4** | Gender mix mismatch (impostors forced to 50/50, targets followed qualifying pool). | Matched impostor gender counts ($N_{\text{imp}, M}, N_{\text{imp}, F}$) exactly to target gender counts ($N_{\text{tgt}, M}, N_{\text{tgt}, F}$) per partition. Single-chapter speakers excluded symmetrically from targets and impostors. | `src/data/trial_generator.py`<br>(lines 226–280) | `tests/test_prefix_review_failures.py::test_h4_impostor_gender_counts_must_match_target_gender_counts`<br>`tests/test_data_integrity.py::test_trial_generator_strict_cross_chapter_and_gender_match` |
| **H5** | Target vs impostor sampling discrepancy (chapter-uniform vs cross-chapter utterance weighted), causing duration shortcut. | Replaced chapter-uniform impostor sampling with cross-chapter pair-weighted sampling matching target marginals. Excluded single-chapter speakers from both classes. Added hard gate asserting relative duration difference $< 3\%$. Logged mean and std durations per class. Measured: 0.33% diff on `test-clean`, 2.48% on `test-other`, 1.93% on `dev-clean`. | `src/data/trial_generator.py`<br>(lines 211–225, 292–309) | `tests/test_prefix_review_failures.py::test_h5_mean_durations_target_vs_impostor_differ_by_less_than_3_percent`<br>`tests/test_data_integrity.py::test_trial_generator_strict_cross_chapter_and_gender_match` |

---

## 2. Medium & Minor Findings Resolution Table

| Finding | Description | Resolution & Implementation Details | Modified Files | Verifying Test(s) |
| :--- | :--- | :--- | :--- | :--- |
| **Silent impostor shortfall** | If `max_attempts` was reached, fewer impostors were returned silently. | Added hard check: raises `RuntimeError` if required impostor count cannot be produced within `max_attempts = needed * 150`. | `src/data/trial_generator.py` (lines 274–278) | Tested via mock generator assertions |
| **`max_total_trials` truncation** | Truncated trials after shuffling, breaking 1:1 balance and contradicting report. | Fixed truncation: splits `max_total_trials` equally across targets and impostors (`half = max_total_trials // 2`), preserving exact 1:1 balance. | `src/data/trial_generator.py` (lines 282–286) | Tested via synthetic pipeline test |
| **Nominal target uniformity** | Target count allowed silent shortfall for small pair pools; dead `median_pairs` code. | Removed `median_pairs` dead code. Enforced `targets_per_speaker=100` with hard assertion raising `RuntimeError` if any qualifying speaker has fewer pairs. | `src/data/trial_generator.py` (lines 174–185) | `tests/test_data_integrity.py::test_trial_generator_strict_cross_chapter_and_gender_match` |
| **Reuse capping & ADR claim** | ADR-016 claimed reuse was capped at 4–5x, which held only for mean reuse. Max reuse reached 23–40x due to LibriSpeech chapter-size asymmetry. | Documented mathematical impossibility of hard cap $\le 10$ given LibriSpeech chapter asymmetries (ADR-017). Exported full reuse percentiles (`p25`, `p50`, `p75`, `p90`, `p95`, `p99`) and histogram in `trials_report_*.json`. | `src/data/trial_generator.py`<br>`docs/DECISIONS.md` (ADR-017) | Tested via JSON report structure assertions |
| **Silent drops & determinism** | UNKNOWN gender dropped silently; `self.rng` shared across calls; row-order dependency. | Tracked and reported UNKNOWN gender drops explicitly in `trials_report.json`. Deterministically sorted manifest rows by `utt_id` upon loading. Instantiated fresh local `random.Random(seed)` per call. | `src/data/trial_generator.py` (lines 78, 88–99, 152) | `tests/test_data_integrity.py::test_trial_generator_strict_cross_chapter_and_gender_match` |
| **Input validation in metrics** | Missing checks for NaNs/Infs, non-binary labels, empty classes in DET curve, and `bfloat16` tensors. | Added comprehensive `_validate_inputs` function: rejects NaNs/Infs (`ValueError`), checks binary $\{0,1\}$ labels (`ValueError`), validates non-empty positive/negative classes in both EER and `compute_det_curve`, and automatically casts `bfloat16`/`float16` to `float32`. | `src/eval/metrics.py` (lines 53–78) | `tests/test_metrics.py::test_input_validation_edge_cases` |
| **Missing DET endpoints** | Missing accept-all (FAR=1, FRR=0) and reject-all (FAR=0, FRR=1) endpoints. | Added explicit endpoints: prepends threshold $-\infty$ (FAR=1.0, FRR=0.0) and appends $+\infty$ (FAR=0.0, FRR=1.0). | `src/eval/metrics.py` (lines 330–345) | `tests/test_metrics.py::test_det_curve_structure_and_endpoints` |
| **minDCF $P=0.01$ statistical power** | With 2,900 impostors, $P=0.01$ has few false alarms; bootstrap CIs are wide. | Designated $P_{\text{target}}=0.05$ as primary benchmark metric and $P=0.01$ as secondary indicative only. Documented convention in `docs/RESULTS_LOG.md` and `docs/EXPERIMENT_PLAN.md`. | `src/eval/metrics.py`<br>`docs/RESULTS_LOG.md` | `tests/test_prefix_review_failures.py::test_h2_mindcf_is_normalized_by_default_with_named_keys` |
| **Bootstrap output completeness** | Missing point estimates, B-side minDCF CIs, and tracking of skipped degenerate resamples. | Returned complete dictionary structure including `point_estimate_a`, `point_estimate_b`, `bootstrap_mean_*`, `ci_95_*` for both systems, `diff_point_estimate`, `diff_ci_95`, and `n_skipped_degenerate`. | `src/eval/metrics.py` (lines 500–544) | `tests/test_metrics.py::test_speaker_bootstrap_ci` |
| **Tightened metric tests** | Analytic Gaussian test used loose $\pm 0.005$ tolerance; SpeechBrain checked only 1 seed. | Tightened Gaussian analytic test to $3\times \text{SE}$ sampling tolerance ($N=100,000$). Extended SpeechBrain parity test across 5 distinct random seeds (`[42, 123, 777, 2025, 9999]`). | `tests/test_metrics.py` | `tests/test_metrics.py::test_analytic_gaussian_eer`<br>`tests/test_metrics.py::test_speechbrain_metric_stats_parity_across_seeds` |
| **Metadata CSV Export** | Callers had to guess trial speaker IDs to perform clustered bootstrap. | Generates companion `trials_<split>_meta.csv` containing `spk_a`, `spk_b`, `gender`, `utt_a`, `utt_b`, `dur_a`, `dur_b` for every trial row. | `src/data/trial_generator.py` (lines 405–412) | Verified across all generated partitions |

---

## 3. Documentation & Historical Log Corrections

1. **Excluded Speaker Discrepancy:**
   - Corrected excluded-speaker lists in `docs/RESULTS_LOG.md` and `docs/DECISIONS.md` to match measured audit reports:
     - `test-clean` (11 speakers): 1188, 2094, 2300, 3575, 3729, 4507, 5639, 672, 7729, 8230, 8455.
     - `test-other` (4 speakers): 1688, 5764, 6938, 8280.
     - `dev-clean` (9 speakers): 1919, 2078, 3000, 3081, 3170, 3576, 3853, 422, 777.
2. **Gender Split in `test-other`:**
   - Corrected raw manifest split from inaccurate "17 M / 16 F" to measured 17 F / 16 M. After excluding 2 F and 2 M single-chapter speakers, qualifying pool is 15 F / 14 M (29 speakers).
3. **External Reference Clarification:**
   - Updated `docs/RESULTS_LOG.md` to state that ECAPA performance is a plausibility check verifying verification mechanics and bounds, not proof of zero leakage or shortcuts.
4. **Statistical Power & Cohort Size:**
   - Documented that with $N=29$ qualifying speakers per test partition, detectable differences are subject to speaker-level sampling variance, necessitating multi-seed runs and speaker-level difference CIs.


# Review: `trial_generator.py` and `metrics.py`

I made no edits. I re-read both files and checked the generated trial files and manifests. I also found errors in my own earlier summary (last section).

## Priority findings

| # | File | Finding |
|---|---|---|
| **H1** | `metrics.py` L67-72 | **The label/threshold-direction check does nothing.** The body of `if check_direction and mean_pos < mean_neg:` is `pass`. The docstring says it "asserts/warns", and you asked for failing loudly. Inverted scores or labels return an EER such as 0.97 with no error. `test_label_direction_inversion` only checks the arithmetic identity (`EER + EER_flipped ≈ 1`). It bypasses the check with `check_direction=False`, so it never tests loud failure. |
| **H2** | `metrics.py` L101-133 | **minDCF is unnormalized by default (`normalize=False`), and the logs don't say so.** The values logged for the ECAPA reference (0.0016 at 0.01, 0.0053 at 0.05) are raw `C_det`. Normalized by `C_def`, they would be 0.16 and 0.106. The paper's 0.043 at 0.65% EER (OPEN_QUESTIONS Q4) is plausibly a normalized value. Raw and normalized numbers must not be compared. `RESULTS_LOG.md` should state which convention is used. |
| **H3** | `metrics.py` `paired_bootstrap_ci` | **The cluster resampling is statistically too narrow.** Each trial gets one cluster ID, but an impostor trial depends on two speakers. Utterances are heavily reused (max reuse 23 in test-clean, 29 in test-other, 40 in dev-clean), so trials share utterances. There is also no helper that defines the cluster, so the caller has to guess (enrollment speaker? target speaker?). Expect CIs that are too optimistic unless resampling is done over speakers and utterances, or the design is documented. The CI also covers speaker variance only, not training-seed variance. |
| **H4** | `trial_generator.py` L185-197 | **Target and impostor gender mix differ.** Impostors are forced to 50/50 M/F, while targets follow whichever speakers qualify. Measured: test-clean targets are 1300 M / 1600 F but impostors 1450 / 1450. test-other targets are 1400 M / 1500 F, and dev-clean targets are 1500 M / 1600 F. Cosine score distributions differ by gender, so this adds a small, systematic class-conditional shift. Impostor gender counts should be matched to the target gender counts. |
| **H5** | `trial_generator.py` L214-217 vs L175-181 | **Target and impostor utterances are sampled differently.** Targets are drawn utterance-weighted from cross-chapter pairs. Impostors pick a chapter uniformly, then an utterance. Mean utterance duration is therefore different per class: test-clean 7.80 s for targets vs 8.70 s for impostors, dev-clean 7.41 vs 7.77, test-other 7.08 vs 7.14. Duration affects cosine scores, so this is a weak shortcut. It is largest on test-clean, the set the ECAPA reference was logged on. |

## Medium findings

- **Silent impostor shortfall** (`trial_generator.py` L208-225). If `max_attempts` is hit, fewer impostors are returned with no error or warning, which breaks the claimed 1:1 balance.
- **`max_total_trials` truncation** (L234-235). It cuts after shuffling, so target/impostor balance and per-speaker uniformity are lost. The report's target and impostor counts are computed before truncation, so they would contradict `total_trials_count`. This path wasn't used in the runs.
- **Target count is only nominally uniform** (L160-181). `targets_per_speaker = min(100, max(20, min_pairs))` and `k = min(...)` silently give a speaker fewer targets if its pair pool is small. `median_pairs` is dead code. It happened to be exactly 100 per speaker on these three sets, but nothing enforces it.
- **Reuse is not capped.** The ADR says reuse stays "under ~4-5x". That is true of the mean only. Per-utterance reuse reaches 23 to 40, which makes trials strongly correlated (see H3).
- **Silent drops in `generate_trials`.** A speaker with `gender == "UNKNOWN"` is dropped from both impostor pools silently. Two further points affect reproducibility. `generate_trials` consumes the shared `self.rng`, so calling it twice on one object gives different lists. Output also depends on manifest row order.
- **No input validation in `metrics.py`.**
  - NaN or inf scores are not rejected. `np.unique`, `sort` and `searchsorted` then give silently wrong EER.
  - Labels outside {0,1} are silently ignored.
  - `compute_det_curve` skips the empty-class check, so it returns NaN arrays.
  - A bfloat16 tensor fails at `.numpy()`.
- **Operating-point resolution** (`metrics.py` L92-96). EER is the average of FAR and FRR at the single threshold with the smallest gap. It is not interpolated. The error is at most about 1/n_trials, which is negligible at these sizes. Ties pick the lowest threshold.
- **Missing endpoint.** The candidate thresholds exclude the "accept-all" point (FAR=1, FRR=0). That is harmless for EER and minDCF but leaves the DET curve incomplete. SpeechBrain behaves the same way.
- **minDCF at P=0.01 is statistically thin.** With about 2,900 impostor trials, the optimum sits where only a handful of impostor trials are false accepts. Bootstrap CIs on it will be wide and biased, so treat it as indicative only.
- **Bootstrap output details.** The function returns the bootstrap mean (`eer_a_mean`) but never the point estimate. Degenerate resamples are skipped without a warning, and only `n_valid_bootstrap` shows it. No B-side minDCF CIs are returned, only differences.
- **Unchecked test claims.** `test_analytic_gaussian_eer` allows ±0.005 against a sampling SE of about 0.0016. `test_speechbrain_metric_stats_parity` checks only one Gaussian seed.

## What I checked and found fine

- **Disjoint chapters.** No chapter ID is shared across speakers in any manifest, so impostors are always cross-chapter too. There is no same-chapter shortcut for targets.
- **Strict cross-chapter targets.** Nothing in the code falls back to same-chapter pairs.
- **Duration filter order.** It runs before the single-chapter classification, which is correct.
- **Pairing and direction.** Accept is `score > threshold`, FRR counts targets at or below it, and FAR counts impostors above it. Both match SpeechBrain, and the parity test confirms it. The bootstrap uses the same resample indices for A and B within a call.

## Corrections to my earlier Phase 1B summary

1. **I listed excluded speaker IDs for test-clean and test-other that I had not read from the reports.** The IDs I gave for those two sets were wrong. I only printed the counts at the time. The reports give:
   - test-clean: 1188, 2094, 2300, 3575, 3729, 4507, 5639, 672, 7729, 8230, 8455
   - test-other: 1688, 5764, 6938, 8280

   The dev-clean list I gave was correct.
2. **test-other has 17 F / 16 M, not 17 M / 16 F.** I repeated the docs' unverified figure as if it were measured. The train (126 M / 125 F), dev-clean and test-clean (20/20) splits do match the manifests.
3. **`RESULTS_LOG.md` overclaims.** The ECAPA entry says the EER range check "prov[es]" there is no leakage or shortcut. A plausibility range proves no such thing, and H4 and H5 are real, if small, shortcuts. I also called the metrics tests "100% coverage" without measuring it.
4. **The test-other EER (1.79%) is lower than test-clean (2.52%), which I did not flag.** No CIs were reported, and with 29 qualifying speakers per set the difference may be within noise. It also needs checking against H4 and H5, since test-clean has the largest duration gap.

## Decision for you

If you want fixes, I suggest this order: H1, H2, H4 and H5, then H3. Regenerating trial lists (H4 and H5) will change the SHA-256 hashes and the logged reference numbers. I would re-run the ECAPA reference and log it as a new run rather than overwrite the old one.
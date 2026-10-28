Review of the SKA-HNN / IAAMSoftmax PRD
I checked the PRD against the paper PDF and did some back-of-envelope arithmetic on the parameter counts. The PRD is strong on structure, but it has several numeric and provenance problems that would hurt you in a viva. I could not verify the external-source claims (companion papers, GitHub repos) from the materials here, so I flag those as unverified rather than wrong.
What's good
* The tier structure with fallbacks and go/no-go gates is practical.
* The cross-chapter and gender-matched trial design is a real methodological strength.
* Correcting Eq. 3 to j ≠ y_i is right.
* The ablation switches (S0–S6) match the paper's definitions of S2, S3, S5 and S6.
* The margin warm-up, the min_keep_ratio guard and the metric unit test against SpeechBrain are sensible.
* The scaling approach (keep the topology, shrink the width) is the right call.
Critical issues
1. Parameter counts are not trustworthy, and the "paper fact" labels hide that.
* The paper's Table VIII gives only dimensions, not per-layer parameters. The PRD's parameter column is labeled "Paper Fact + Engineering Verification" but is the author's own calculation.
* The MLASP row (8.81M) uses d_attn=614, while Section 2.2 and the code defaults use 244–256. At 256 the attention block is about 3.7M, and the full model would come out near 25.7M, not 30.91M. The 614 looks reverse-engineered to hit the total.
* ResBlock is listed at 1.334M. The PRD's own code (5 convs of 128→128, 3×3) gives about 0.75M.
* The paper's S1→S5 delta is 13.5M parameters. SKA branch convs in your reconstruction can't account for that. Either the paper's front-end differs from your reconstruction, or S5 changes width. That belongs in the ambiguity table, not under "completely consistent."
* My estimate for the scaled model (C=32, C1=256, attn 128) is about 4.1–4.2M, not 2.65M. The S3/S5/S6 counts (1.71M, 2.41M, 1.54M) look wrong too. My S6 estimate is about 2.1M.
* This breaks the "11.6× fewer parameters" claim and the Tier 0 baseline (1.95M) is not parameter-matched to your model. That is the same confound you criticize in the paper's Table II.
* Fix: run torchinfo first, then rewrite every parameter figure from measured values.
2. Expected results are written as if they were results.
* Viva Q10 asks "why does your implementation achieve ~2.2% EER", before anything has been run.
* Q7 states that memorization starts around epoch 15–20. Q8 states that clean embeddings sit within 55° while mislabeled ones sit at 75–90°. Q9 gives a 15–30% noise rate for CAMSAT. Q1 says fwSKA picks 3×3 for low-frequency pitch bands and 5×5 for formants.
* None of these come from the paper. They are plausible narratives, and an examiner who asks "where did you get that?" will catch you out.
* Go/no-go gates like "GO if removing SKA/MLFC/2D-CNN degrades EER" turn hoped-for outcomes into pass criteria. At 251 speakers and a single seed, the S2/S3/S5 differences may not be monotonic. Report what you get.
3. The scaling-law "defense" for the EER gap is invalid.
* The regression EER ≈ 9.15 − 1.08 ln N predicts a negative EER at N=5994 (about −0.24%).
* It also disagrees with the numbers the PRD quotes alongside it (1.87% at N=500; the formula gives about 2.4%).
* It was fit on LibriSpeech, so it can't explain a cross-dataset gap to VoxCeleb1-O. The "ΔEER ≈ 3.43%" is also larger than the actual gap it's meant to explain (~1.5%).
* Present the gap qualitatively, as fewer speakers, a different domain, and a smaller model.
4. Tier 3 SSL design may leak labels or fail.
* "K-Means on bootstrap embeddings" is only self-supervised if the embeddings come from an unsupervised source. If they come from a Tier 0 model trained with true labels, that's leakage.
* K-Means on MFCC statistics will largely cluster by channel or chapter, not speaker, so labels may be far noisier than useful.
* Report pseudo-label purity and NMI against ground truth.
* Also note that a ~4M-parameter model may not memorize noisy labels in 30 epochs. AAMSoftmax may then degrade less than you hope, so the "≥10–25% gain" target could be unrealistic.
5. AdaptiveDrop defaults conflict with the paper's logic.
* The PRD sets ESC=7 and ESD=5, so filtering starts before correction. That contradicts the paper's stated rationale of correcting first to avoid over-cleaning. It also contradicts the paper's Algorithm 1, which has correction at epoch ≥ ESC, filtering at epoch ≥ ESD, and a 5-epoch warm-up.
* The quoted "ESD = 5, ESC = 7" and the correction rule (τ_corr = 0.5) are attributed to companion papers I can't verify. Check the original before citing.
* On clean data, a fixed τ=0.423 will drop hard clean samples. Your "keep ratio > 85%" gate may fail.
Moderate issues
* Timing and memory are unmeasured estimates presented as fact (55 s/epoch, 3.8 GB, "same on T4 and M3"). Free Colab has about 2 CPU cores, so decoding and augmenting 28.5k FLACs per epoch will likely dominate. Caching all of train-clean-100 as int16 is about 11.6 GB, which doesn't fit in Colab RAM. MPS also lacks the same fp16 and GradScaler support, and your autocast code is CUDA-only.
* fp16 local statistics: E[x²] − μ² in half precision loses precision badly. Compute the local statistics in fp32.
* Trial lists:
    * Many LibriSpeech speakers have only one chapter, and your code silently falls back to same-chapter pairs for them. Either drop those targets or report the fraction.
    * Bootstrapping over trials understates uncertainty, because 30k trials share 40 speakers and a few thousand utterances. Use paired bootstrap over utterances or speakers, with the same resamples for each system.
    * Use ≥3 seeds for the headline comparisons (S0 vs S1, noise).
* minDCF prior: the paper doesn't state P_target. Its minDCF values (e.g., 0.043 at 0.65% EER) look more like P=0.05 than 0.01. Check before comparing.
* Demo expectations: "cosine > 0.65 across different sentences" and an impostor range of [−0.05, 0.25] are unlikely for a small LibriSpeech-trained model on a laptop mic. Calibrate from data and don't promise numbers. MLASP attention is per-channel, so you need to average it to plot a frame curve.
* LibriTTS argument: the claim that 24 kHz "distorts the Mel frequency axis" is wrong, since you can compute 80 Mels to 8 kHz at any rate. Keep the cost and trimmed-silence arguments instead.
Minor issues
* ASP originates with Okabe et al. (2018), not Desplanques 2020, who introduced channel-dependent ASP.
* The SKA-TDNN paper is cited as SLT 2022 in places, but the paper's reference [24] says SLT 2023.
* The readourlips/SKA-TDNN repo URL looks suspect.
* Appendix 12.2 rows 3, 5 and 6 are truncated mid-LaTeX.
* Paths are inconsistent: scripts/prep_librispeech.py vs src/data/prep_librispeech.py.
* The LS equation uses shrinking neighborhoods, while the code uses replicate padding.
* test-other is only a loose analogue of VoxCeleb1-H.
Suggested order of work
1. Implement the model, run torchinfo, and replace all parameter numbers. Decide the Tier 0 baseline width to match.
2. Benchmark real epoch time (data loading included) on your actual hardware, then re-scope the number of runs.
3. Delete or relabel every viva answer and table cell that states an unrun result. Phrase them as hypotheses.
4. Fix the Tier 3 pseudo-label source and add purity metrics.
5. Revisit ESC/ESD ordering against the original AdaptiveDrop paper.
I can draft a corrected parameter-audit script or a leaner viva Q&A that separates paper-stated claims 
---
description: Always-on scientific integrity and non-hallucination rules
alwaysApply: true
---

# Scientific Integrity & Non-Hallucination Rules

1. **Never State Unrun Numbers as Facts:**
   - Any number claimed as an experimental result must be measured from an executed model run and recorded in `docs/RESULTS_LOG.md` with an active `run_id`.
   - Any unrun, literature, or anticipated number must be explicitly labeled `[HYPOTHESIS]`.
2. **Never Hand-Calculate Parameter Counts:**
   - Layer and total parameter counts must originate strictly from `torchinfo.summary()` execution via `.agents/workflows/param-audit.md`.
   - Until measured, all parameter counts in documentation must be written as `"TBD: measure with torchinfo"`.
3. **Mandatory Provenance Tags:**
   - Every architectural detail, equation, hyperparameter, and empirical claim must carry one of:
     - `[Paper]`: Verbatim or mathematically direct from Alam et al. (IJCNN 2025).
     - `[External-UNVERIFIED]`: Derived from external papers, theses, or external repos. Must be logged in `docs/OPEN_QUESTIONS.md`.
     - `[Engineering Choice]`: Deliberate design or scaling decision logged in `docs/DECISIONS.md`.
4. **Technical-Only Go/No-Go Gates:**
   - Progression gates must be purely technical: zero NaNs, exact tensor shapes matching contracts, passing unit tests, and non-zero gradients.
   - Gates must NEVER demand "result must improve" or "EER must decrease".
5. **Faithful Reporting of Negative Results:**
   - Non-monotonic ablations, null gains, or degraded metrics must be reported immediately and logged as-is without massaging or retrying.
6. **When Unsure, Ask:**
   - Do not invent paper details or fill in blanks with plausible-sounding numbers. Ask the user or record in `docs/OPEN_QUESTIONS.md`.

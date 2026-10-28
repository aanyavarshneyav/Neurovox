# Workflow: /pre-commit-check

**Purpose:** Comprehensive pre-commit sanity check verifying that tests pass, code adheres to style and numerics guidelines, documentation links are valid, and `AGENTS.md` remains strictly under 11,000 characters.

---

## Step-by-Step Procedure

1. **Verify `AGENTS.md` Size Constraint:**
   ```bash
   chars=$(wc -c < AGENTS.md | tr -d ' ')
   echo "AGENTS.md size: $chars chars (must be < 11,000)"
   if [ "$chars" -ge 11000 ]; then
     echo "ERROR: AGENTS.md exceeds 11,000 character limit!"
     exit 1
   fi
   ```

2. **Run Pytest Suite:**
   ```bash
   pytest tests/ -v
   ```
   Ensure 100% passing tests (unit tests for tensor shapes, local statistics numerics, trial generator cross-chapter assertion, loss clamping, metric computations).

3. **Check for Unmeasured Hallucinations / Numbers:**
   - Run grep across `docs/` to ensure no raw numeric results or parameter counts exist without `[HYPOTHESIS]`, `[Paper]`, or `TBD: measure with torchinfo`:
     ```bash
     grep -rnE "(achieves|yields|parameters:)[ ]*[0-9]+(\.[0-9]+)?" docs/ | grep -vE "(TBD|HYPOTHESIS|Paper|Table|ref|IEEE|202[0-9]|192|128|512|32|256|80|180|16000|0\.423|0\.2|30)"
     ```

4. **Verify Documentation Updates:**
   - If code changes introduced new configuration parameters or architectural choices, verify that `docs/DECISIONS.md`, `docs/PROJECT_SPEC.md`, or `docs/ARCHITECTURE.md` were appropriately updated.

5. **Review Git Status:**
   ```bash
   git status --short
   ```
   Confirm no temporary files, datasets (`data/`), model checkpoints (`checkpoints/`), or virtual environments (`.venv`) are staged for commit.

6. **Request Explicit User Permission Before Committing:**
   - Present the proposed commit message and list of staged files to the user.
   - **NEVER execute `git commit` without asking for and receiving explicit user permission first.**


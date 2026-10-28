# Workflow: /session-handoff

**Purpose:** Compile an atomic summary of current repository state, immediate next tasks, and active technical risks into `docs/HANDOFF.md` at the conclusion of an engineering session.

---

## Step-by-Step Procedure

1. **Audit Current Repository State:**
   - Run `git status` and `git log -n 3` to identify recent commits and modified files.
   - Run `pytest tests/` to confirm test suite status.

2. **Summarize What Was Accomplished:**
   - Catalog the key artifacts, modules, or documentation generated during the session.

3. **Identify the Next 3 Actionable Tasks:**
   - State the top 3 concrete tasks in execution order, with clear input and output deliverables.

4. **Identify Active Technical Risks & Blockers:**
   - Note any open questions, hardware bottlenecks, numerical concerns, or external dependencies.

5. **Write `docs/HANDOFF.md`:**
   Generate or overwrite `docs/HANDOFF.md` using the following standardized schema:
   ```markdown
   # Session Handoff & State Summary
   
   **Date:** YYYY-MM-DD  
   **Current Phase:** Phase X  
   **Git Commit:** <hash>  
   **Test Status:** All tests passing (<N> passed)
   
   ## 1. Accomplishments This Session
   - ...
   
   ## 2. Next 3 Immediate Tasks
   1. ...
   2. ...
   3. ...
   
   ## 3. Active Risks & Notes for Next Agent
   - ...
   ```

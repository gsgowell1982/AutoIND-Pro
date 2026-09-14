# Evidence-Complete Study Panel Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve r2 page-85 study metadata and its complete table-note group without weakening correct table ownership elsewhere.

**Architecture:** Treat visual-panel ownership transfer as a validated commit. Recover missing row geometry from word evidence before cropping the source table, and scope Markdown note identity to its owner/group rather than normalized text alone.

**Tech Stack:** Python, unittest, PyMuPDF-derived word evidence, existing PDF AST and Markdown projection helpers.

---

### Task 1: Add failing ownership and Markdown contracts

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Add a deterministic Markdown normalization test with two tables that own separate `附加信息：` note groups.
- [ ] Strengthen the real r2 page-85 test to require a valid template bbox, AST presence, and column projection.
- [ ] Add a real r2 Markdown ordering assertion for the page-85 label and its two continuation lines.
- [ ] Run only these tests and confirm failures in the missing-label and missing-template paths.

### Task 2: Make visual-panel transfer evidence-complete

**Files:**
- Modify: `parsers/pdf/postprocess.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Add a helper that clusters table word evidence into stable physical row records.
- [ ] Use word-row geometry when visual cells contain text but no bbox.
- [ ] Pass source-aware row records into study metadata template construction.
- [ ] Reject a destination template that cannot be synchronized into AST.
- [ ] Prevent source-table cropping when no committable destination exists.
- [ ] Run the page-85 template tests and confirm they pass.

### Task 3: Scope note deduplication to semantic identity

**Files:**
- Modify: `api/main.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Classify grouped note segments and pure note-group labels as repeatable across distinct tables.
- [ ] Keep existing local exact-segment deduplication unchanged.
- [ ] Run deterministic and real r2 Markdown note tests and confirm both labels render.

### Task 4: Focused verification and state sync

**Files:**
- Modify: `D:/ind-session/ENGINEERING_DECISIONS.md`
- Modify: `D:/ind-session/ACTIVE_WORK_CONTEXT.md`
- Modify: `D:/ind-session/CURRENT_CODEX_STATE.yaml`

- [ ] Run page 84-86 study-panel, table, note ownership, and Markdown regressions.
- [ ] Run related deterministic Markdown/export tests.
- [ ] Run `py_compile` and `git diff --check` on touched code and tests.
- [ ] Synchronize touched project files to the governed mirror and verify hashes.
- [ ] Record the evidence-complete transfer and note-group identity decision in session state.

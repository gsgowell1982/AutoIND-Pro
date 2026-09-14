# Overview Parent Header Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent a multilevel overview-table parent header from rendering again as standalone text.

**Architecture:** Carry parent-header geometry from word projection into the overview semantic projection. Extend the existing ownership closure to match AST text against explicit semantic-header regions and record source ownership without expanding the physical table box.

**Tech Stack:** Python, PyMuPDF-derived AST geometry, `unittest`, IND-review Markdown.

---

### Task 1: Add Failing Ownership Regressions

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [x] Add a focused test that supplies a semantic parent-header bbox above the table and expects only the overlapping same-text AST block to be owned.
- [x] Add a page-77 end-to-end test requiring `txt_p77_061` to be owned by `tbl_021` and removed from standalone AST flow after reconciliation.
- [x] Require the page-77 Markdown region to contain the multilevel table header and no standalone `位置` line.
- [x] Run both tests and confirm failure from missing semantic-header ownership.

### Task 2: Preserve And Close Parent-Header Ownership

**Files:**
- Modify: `parsers/pdf/postprocess.py`

- [x] Return overview parent-header groups with their source geometry from word projection.
- [x] Pass those groups into overview projection instead of reconstructing provenance-free groups from strings.
- [x] Match AST text blocks against semantic header-group text and bbox during table ownership closure.
- [x] Record source IDs on the table, header group, and span header cell; mark the AST block as table-owned metadata.
- [x] Run the focused and page-77 tests and confirm they pass.

### Task 3: Verify Affected Scope

**Files:**
- Verify: `parsers/pdf/postprocess.py`
- Verify: `tests/parser_tests/test_r2_regression.py`

- [x] Run page-77 and nearby overview-inventory regressions.
- [x] Synchronize the same focused changes to the maintenance mirror without overwriting unrelated work.
- [x] Run `py_compile`, scoped `git diff --check`, document review, and SHA-256 parity checks.
- [x] Do not commit, merge, push, reset, or clean either worktree.

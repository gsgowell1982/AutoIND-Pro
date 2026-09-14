# Repeated Cross-Page Note Occurrence Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve both adjacent-page occurrences of `a-总放射性；回收率，14C` and attach each to its correct preceding result-matrix table.

**Architecture:** Select cross-page owners using source occurrence identity and marker-anchor evidence, then suppress only the physical source occurrence committed to that owner. Keep the table projection pipeline unchanged and normalize ordered note groups only at the ownership/render boundary.

**Tech Stack:** Python, PyMuPDF-backed parser AST, Markdown projection, unittest regression tests.

---

### Task 1: Establish the failing two-occurrence contract

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Replace the page 87 table non-ownership assertion with an assertion that it owns the physical page 88 occurrence.
- [ ] Assert the page 86 and page 87 tables each own one distinct occurrence.
- [ ] Assert Markdown renders both occurrences once in their respective table-to-next-object regions.
- [ ] Run the focused tests and confirm failure because page 87 has no note ownership.

### Task 2: Select the semantic previous-table owner

**Files:**
- Modify: `parsers/pdf/postprocess.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Extract markers from the page-top note candidate.
- [ ] Permit the last preceding result-matrix table when its cells contain a compatible terminal marker, even if its raw bbox misses the page-bottom threshold.
- [ ] Preserve the existing near-bottom and continuation-state path for unmarked statistical notes.
- [ ] Attach the physical page and owner table id to the committed note.

### Task 3: Suppress only committed source occurrences

**Files:**
- Modify: `parsers/pdf/postprocess.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Remove blanket `note_page + 1` normalized-text suppression.
- [ ] Keep source-id and claimed-page suppression for owned notes.
- [ ] Verify the first occurrence cannot suppress an equal second occurrence.

### Task 4: Preserve source note order

**Files:**
- Modify: `api/main.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Prefer a source-positioned note label over a source-id-only duplicate for ordering.
- [ ] Render label and continuation as separate lines in label-first order.
- [ ] Confirm no label is synthesized for the page 87 table.

### Task 5: Focused verification and context synchronization

**Files:**
- Modify: `D:/ind-session/ENGINEERING_DECISIONS.md`
- Modify: `D:/ind-session/ACTIVE_WORK_CONTEXT.md`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Run the two-occurrence regression and related page 84-88 note ownership tests.
- [ ] Run shared deterministic Markdown note tests and the page 80 marker-note baseline.
- [ ] Run Python compilation and `git diff --check` on touched files.
- [ ] Record the occurrence-identity and risk-based regression decisions in session context.

# Adjacent Pre-Table Header Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover adjacent study-context fields, wrapped leaf headers, and parent span headers for borderless IND tables and render their semantic grids in Markdown.

**Architecture:** Add a source-geometry pass that operates independently of `study_metadata`, writes the existing pre-table projection contract, and transfers consumed source ownership to the table. Extend Markdown semantic-grid selection to consume that contract directly.

**Tech Stack:** Python, PyMuPDF-derived page words, existing PDF postprocessor, `unittest`, IND-review Markdown renderer.

---

### Task 1: Add RED Regressions

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Add a synthetic Markdown test proving a pre-table projection selects its two-row semantic grid.
- [ ] Add a synthetic adjacent-text test for two compact context fields, one wrapped leaf header, and one two-column parent span.
- [ ] Add real page 81-83 assertions for semantic headers, source ownership, bullet rendering, and absence of leaked standalone labels.
- [ ] Run the focused tests and verify expected failures before production edits.

### Task 2: Recover Adjacent Text Semantics

**Files:**
- Modify: `parsers/pdf/postprocess.py`

- [ ] Derive header-word anchors from page words and the table header band.
- [ ] Classify single-column stacked fragments as wrapped leaf headers and update the logical header text.
- [ ] Classify multi-column labels as parent spans and write `header_column_groups`, `span_header_cells`, and a two-row semantic grid.
- [ ] Transfer consumed source IDs to the table and mark the source blocks as table-owned metadata.
- [ ] Mark two-or-more adjacent colon fields before the header band as separate unmarked list items.

### Task 3: Render Semantic Headers

**Files:**
- Modify: `api/main.py`

- [ ] Treat `pre_table_parent_header_projection` as a direct semantic-grid export capability.
- [ ] Preserve repeated parent labels in Markdown because Markdown has no colspan primitive.
- [ ] Ensure wrapped leaf headers appear once and consumed source fragments do not render separately.

### Task 4: Focused Verification And Mirror Sync

**Files:**
- Verify: `api/main.py`
- Verify: `parsers/pdf/postprocess.py`
- Verify: `tests/parser_tests/test_r2_regression.py`
- Mirror: `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai`

- [ ] Run synthetic projection tests and real page 81-83 regressions.
- [ ] Run page 80-86 study-table and Markdown regressions plus existing grouped-header tests.
- [ ] Audit every r2 adjacent-pre-table projection for source ownership and column bounds.
- [ ] Run `py_compile` and `git diff --check`.
- [ ] Apply scoped patches to the mirror, rerun focused tests, and verify matching hashes.

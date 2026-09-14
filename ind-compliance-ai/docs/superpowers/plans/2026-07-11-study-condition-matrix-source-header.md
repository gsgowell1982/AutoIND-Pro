# Study Condition Matrix Source Header Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render study-condition result matrices in source order with explicit leaf-header and row-axis rows, without synthetic `条件/时间` labels.

**Architecture:** Recover the source stub located on the repeated leaf-header visual row and store two typed matrix header rows in the existing semantic projection. Keep the current result semantic grid for compatibility, while making Markdown consume the typed header rows and descriptor rows in source order.

**Tech Stack:** Python, PyMuPDF-derived geometry evidence, unittest/pytest-compatible regression tests.

---

### Task 1: Establish the failing Markdown behavior

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Add a focused unit test whose projection contains descriptor rows and typed source matrix header rows.
- [ ] Assert that `种属` is first, `排泄途径(4)` precedes a blank `时间` row, and `条件/时间` is absent.
- [ ] Run the focused unit test and confirm it fails against the current renderer.

### Task 2: Recover and expose source matrix header roles

**Files:**
- Modify: `parsers/pdf/postprocess.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Recover the unique stub atom to the left of the repeated leaf headers on the same visual row.
- [ ] Store `matrix_leaf_header` and `matrix_row_axis` records in the binding and projection.
- [ ] Add page 86 assertions for the recovered source text, row roles, and column widths.

### Task 3: Render source order

**Files:**
- Modify: `api/main.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Replace the synthetic descriptor-branch `条件/时间` row with typed projection rows.
- [ ] Emit descriptor rows first, followed by the leaf-header row, blank row-axis row, and result rows.
- [ ] Preserve the existing fallback when typed rows are unavailable.
- [ ] Run the focused unit and page 86 tests until green.

### Task 4: Focused impact verification

**Files:**
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Verify page 87 receives the same source-backed behavior.
- [ ] Run page 85-87 ownership, binding, colspan, and Markdown regression tests.
- [ ] Run `py_compile` for modified production modules and `git diff --check`.
- [ ] Inspect the final page 86 Markdown region and confirm source order manually.

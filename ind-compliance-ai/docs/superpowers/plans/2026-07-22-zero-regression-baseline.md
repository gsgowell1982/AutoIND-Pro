# Zero Regression Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the five current r2 regression failures through one source-order fix and four semantic test-contract corrections.

**Architecture:** Preserve raw-row source order when trailing grid rows become notes, and consume that metadata in the existing float-segment ordering layer. Keep parser-generated identifiers and normalized semantic leaves internal; tests validate ownership and visible source evidence instead of incidental representations.

**Tech Stack:** Python 3, existing PDF AST post-processing, IND-review Markdown renderer, `unittest`.

---

### Task 1: Lock trailing-note lineage and order

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`
- Modify: `parsers/pdf/table_modules/postprocess.py`
- Modify: `api/main.py`

- [ ] Add a unit test with a trailing raw-grid note and a positioned below-table continuation. Assert raw-row identity and trailing-before-below ordering.
- [ ] Run the unit test and confirm RED because source-order metadata is absent and ordering is reversed.
- [ ] Add raw-row lineage, physical page, and table-boundary source-order coordinates in `extract_trailing_table_note_rows`.
- [ ] Extend `_markdown_table_float_segment_order_key` to use bbox coordinates first and explicit source-order coordinates second.
- [ ] Run the unit test and page 24 regression; confirm GREEN and complete `-未检测` output.

### Task 2: Replace ordinal architecture assertions

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Resolve every `source_object_id` through `structure_templates`.
- [ ] Assert absorbed ownership, owner-table membership, source reasons, and source pages instead of `structure_template_053/054/057` literals.
- [ ] Run both architecture tests and confirm GREEN.

### Task 3: Align presentation assertions with source evidence

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Change page 37 to require a literal `附加信息：` line before the remark and reject a list-prefixed label.
- [ ] Change page 101 to require the grouped dose row and animal-count M/F evidence while rejecting an independent sex row when source evidence is false.
- [ ] Run both tests and deterministic Markdown note/sex-row tests.

### Task 4: Verify, document, and mirror

**Files:**
- Modify: `D:/ind-session/ENGINEERING_DECISIONS.md`
- Synchronize scoped files to the configured mirror.

- [ ] Run syntax compilation and `git diff --check`.
- [ ] Run the five-test former-baseline set and relevant page 24/37/101/106/109/114 tests.
- [ ] Run the complete `tests.parser_tests.test_r2_regression` module and require zero failures.
- [ ] Record the source-order lineage and stable-test-contract decision.
- [ ] Copy scoped code, tests, spec, and plan to the mirror; compare SHA-256 and run focused mirror tests.


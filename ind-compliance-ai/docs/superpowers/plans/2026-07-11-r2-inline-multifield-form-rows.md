# R2 Inline Multifield Form Rows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve same-physical-row field groups in blank IND form templates and render each group as one Markdown bullet.

**Architecture:** Build an `inline_form_row_projection` from owned field text nodes, retaining source IDs, bboxes, visual rows, and inferred column indexes while leaving individual `fields` intact. Markdown replaces only projection-consumed flat rows with ordered row display text; matrix labels, notes, titles, and continuation ownership remain unchanged.

**Tech Stack:** Python, `unittest`, PyMuPDF-backed parser geometry, Markdown export.

---

### Task 1: Define projection and rendering contracts

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Add a pure projection test with three same-row fields, two next-row fields, and `M:/F:` section labels that must be excluded.
- [ ] Add a Markdown helper test requiring grouped display rows to replace consumed flat rows once without changing later matrix rows.
- [ ] Add a page 54-55 regression requiring two three-field rows, two two-field continuation rows, retained source/bbox/column evidence, and exact Markdown bullets.
- [ ] Run the focused tests and verify RED because `inline_form_row_projection` is absent and Markdown still emits one bullet per field.

### Task 2: Build inline form row projection

**Files:**
- Modify: `parsers/pdf/postprocess.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Select owned text nodes by field signatures, excluding title, notes, sections, and matrix labels.
- [ ] Cluster selected nodes by vertical overlap and median-height-derived center tolerance, then sort each row by X.
- [ ] Infer reusable column anchors from X positions and assign each cell a `column_index`.
- [ ] Store row/cell text, label, value, source block ID, bbox, page, display text, and consumed signatures under `semantic_projection_v2.inline_form_row_projection`.
- [ ] Enrich matching field entries with source block ID, bbox, visual row index, and column index.
- [ ] Apply the projection after template ownership and continuation expansion but before composite refresh.

### Task 3: Render one physical row per bullet

**Files:**
- Modify: `api/main.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Read ordered projected rows in `_structure_template_markdown_visual_row_texts`.
- [ ] Insert all projected display rows at the first consumed flat field position and skip every consumed occurrence exactly once.
- [ ] Leave unconsumed singleton/matrix rows and note exclusions in their existing order.
- [ ] Run the focused tests until green.

### Task 4: Verify affected IND templates

**Files:**
- Verify: `parsers/pdf/postprocess.py`
- Verify: `api/main.py`
- Verify: `tests/parser_tests/test_r2_regression.py`

- [ ] Compile changed Python files and run focused page 54-55 tests.
- [ ] Run related form regressions for pages 52, 54-56, 58, 61, 64, 67, and 69.
- [ ] Audit Markdown to confirm grouped metadata bullets, singleton preservation, and no `M:/F:` metadata projection.
- [ ] Synchronize source, renderer, tests, and this plan to the mirror; verify SHA-256 equality and rerun focused tests there.

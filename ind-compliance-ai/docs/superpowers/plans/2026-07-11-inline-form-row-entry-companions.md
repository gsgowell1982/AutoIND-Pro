# Inline Form Row Entry Companions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render heterogeneous IND form rows such as `design question + dosing field + study number` as one source-faithful Markdown item.

**Architecture:** Extend the existing `inline_form_row_projection` after field-only rows establish reliable y bands and column anchors. Absorb only owned nodes represented by the template's `entries` or `sections` that share the field row geometry, occupy a distinct learned column, do not overlap field cells, and are present in template `row_texts`; retain their eventual AST role and identify them as `row_companion` cells in the projection.

**Tech Stack:** Python, existing PDF postprocessor, `unittest`, IND-review Markdown renderer.

---

### Task 1: Add RED Regressions

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Add a synthetic projection test with a no-colon entry in the first column and two field cells in the second and third columns.
- [ ] Assert the entry becomes a `row_companion`, participates in `display_text`, and is consumed exactly once.
- [ ] Add an overlapping full-width entry and assert it remains outside the projection.
- [ ] Add a real page-64 test asserting the first projected form row and Markdown item contain all three source texts and no split item remains.
- [ ] Run both tests and verify failures occur because entry companions are not yet projected.

### Task 2: Extend The Existing Projection

**Files:**
- Modify: `parsers/pdf/postprocess.py:2449`

- [ ] Preserve the current field candidate and visual-row discovery as the confidence seed.
- [ ] Build companion candidates only from owned text nodes whose signatures occur in template `entries` or `sections`, with no field-label colon, a `row_texts` signature, a valid bbox, and no title/note/instruction identity. The final `structure_template_entry` role is written later in the pipeline and must not be used as an early prerequisite.
- [ ] Attach a companion only when it vertically matches a seeded row, maps near a learned column anchor, occupies a column unused by that row, and does not horizontally overlap an existing cell.
- [ ] Emit `semantic_cell_role` for field and companion cells, use `inline_mixed_form_row` for heterogeneous rows, and include companion source IDs/signatures in existing consumption metadata.
- [ ] Keep `fields` mutation limited to actual field cells.

### Task 3: Verify And Synchronize

**Files:**
- Verify: `parsers/pdf/postprocess.py`
- Verify: `api/main.py`
- Verify: `tests/parser_tests/test_r2_regression.py`
- Mirror: `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai`

- [ ] Run the synthetic tests, page-64 real regression, existing page 54-55 inline-row regressions, page 64-69 template/lattice regressions, and Markdown note regressions affected by row consumption.
- [ ] Audit all r2 `inline_form_row_projection` instances for newly absorbed companions and confirm every one satisfies geometry and ownership guards.
- [ ] Run `py_compile` and `git diff --check`.
- [ ] Apply the same scoped changes to the mirror, rerun focused tests, and verify SHA-256 equality.

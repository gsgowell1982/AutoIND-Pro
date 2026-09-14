# R2 Page 45 Repeated Sparse Stub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the same-band left stub in repeated sparse IND templates and keep later left-column labels as body rows.

**Architecture:** Cluster header words by physical row around the repeated leaf fragments. Select a unique left-side word row aligned with the leaf header as the stub, offset leaf provenance columns after inserting it, and start body collection below the complete header envelope so later row-axis labels remain body rows.

**Tech Stack:** Python, `unittest`, PyMuPDF-backed parser geometry.

---

### Task 1: Define the regression contract

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Add a unit test in `RuledSparseTemplateGeometryTests` proving that a same-band left candidate is selected ahead of a lower left candidate.
- [ ] Add a unit test proving that a lower candidate alone is not promoted to a same-band stub.
- [ ] Update the page 45 regression to require `排泄途径(4)` as column zero, `时间` and `0-T h` as separate body rows, four 3-column groups, and non-conflicting fragment columns.
- [ ] Run the focused tests and confirm they fail because the current implementation selects `时间`.

### Task 2: Reconcile header and body row roles

**Files:**
- Modify: `parsers/pdf/postprocess.py:7111`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Replace the one-direction lower-window stub lookup with a helper that receives leaf fragments, builds their physical header envelope, and selects one left-side row with vertical overlap or baseline alignment.
- [ ] Return no stub when only lower rows exist; do not promote body content without rowspan evidence.
- [ ] Use the full leaf/stub bbox bottom as `header_bottom_y`, allowing body collection to retain the following `时间` and `0-T h` rows.
- [ ] Offset every repeated leaf fragment column by one after inserting the stub column.
- [ ] Keep group count, repeated period, blank group spans, bbox provenance, and source row consumption metadata.
- [ ] Run the focused tests until green.

### Task 3: Verify the affected surface

**Files:**
- Verify: `parsers/pdf/postprocess.py`
- Verify: `tests/parser_tests/test_r2_regression.py`

- [ ] Compile the changed Python files.
- [ ] Run page 45 plus sparse/multilevel template regressions for pages 36, 41, 45, 49, and 52.
- [ ] Render the page 45 IND-review Markdown and verify physical order and absence of duplicate bullet rows.
- [ ] Audit all r2 repeated sparse projections for unintended role changes.
- [ ] Synchronize the two changed source/test files and this plan to the mirror workspace, verify SHA-256 equality, and rerun the focused tests there.

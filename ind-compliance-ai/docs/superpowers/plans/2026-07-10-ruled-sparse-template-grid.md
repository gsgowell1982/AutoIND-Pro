# Ruled Sparse Template Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize blank multi-column template recognition for flat leaf slots, centered parent anchors, and blank group slots with repeated leaf headers.

**Architecture:** Extend the existing ruled multilevel structure-template projection with a normalized sparse-layout analyzer. Keep the page-41 body-lattice path strongest, select one mutually exclusive sparse mode from geometry, and reuse the existing additive semantic-grid/Markdown pipeline.

**Tech Stack:** Python, PyMuPDF geometry, `unittest`, existing PDF postprocessing and Markdown export helpers.

---

### Task 1: Lock End-to-End Behavior

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Add failing page 45 regression**

Assert `blank_group_slots_plus_repeated_leaf_anchors`, four groups, three leaf columns per group, repeated `尿液/粪便/合计`, the `时间` stub, `0-T h`, Markdown table output, `structure_template`, and no business table.

- [ ] **Step 2: Add failing page 49 regression**

Assert ten logical columns, `centered_parent_anchor_over_leaf_slots`, a `位置` span over `卷/页码`, seven body rows, exact footnote-fragment provenance, Markdown table output, and preserved template ownership.

- [ ] **Step 3: Add failing page 52 regression**

Assert `flat_leaf_slots`, the five visible columns, the blank template body rows and footnote markers, Markdown table output, and preserved template ownership.

- [ ] **Step 4: Run the three tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.parser_tests.test_r2_regression.R2RegressionTests.test_page45_excretion_template_projects_blank_groups_with_repeated_leaf_anchors `
  tests.parser_tests.test_r2_regression.R2RegressionTests.test_page49_toxicology_overview_projects_centered_parent_anchor_grid `
  tests.parser_tests.test_r2_regression.R2RegressionTests.test_page52_drug_substance_template_projects_flat_leaf_grid
```

Expected: all three fail because `semantic_projection_v2` lacks `ruled_multilevel_template_header_projection`.

### Task 2: Lock Geometry Contracts

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Add synthetic title-band exclusion test**

Construct a one-slot title band followed by a five-slot leaf band and assert selection of the five-slot band.

- [ ] **Step 2: Add centered-parent positive and negative tests**

Assert a centered short anchor maps to two adjacent child slots. Assert an off-center anchor and a noncontiguous-child candidate are rejected.

- [ ] **Step 3: Add repeated-period positive and negative tests**

Assert twelve leaves with period three and four aligned group slots are accepted. Assert an inconsistent final period and misaligned group centers are rejected.

- [ ] **Step 4: Run the geometry class and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_r2_regression.RuledSparseTemplateGeometryTests
```

Expected: errors or assertion failures because the new geometry helpers do not exist.

### Task 3: Implement Unified Sparse Geometry

**Files:**
- Modify: `parsers/pdf/postprocess.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Add normalized layout entry point**

Add `_ruled_sparse_template_layout(rows, template_bbox, bands, page_words)` returning `None` or a dictionary containing `column_layout_mode`, `column_slots`, `logical_columns`, `parent_groups`, `body_rows`, `header_fragments`, and `body_fragments`.

- [ ] **Step 2: Add title-band filtering and dominant leaf selection**

Filter bands whose center overlaps the title row bbox, require at least five slots, map words by slot, and rank valid candidates by mapped-column coverage then slot count.

- [ ] **Step 3: Add centered-parent inference**

Map a short parent anchor to a contiguous child range using child centers and union-center tolerance. Emit a visible parent group only when at least two children are supported.

- [ ] **Step 4: Add repeated-leaf/group inference**

Compute the smallest exact repeated normalized period, validate multiplication and group-center alignment, and emit blank group spans without synthetic labels.

- [ ] **Step 5: Add flat fallback**

Accept a unique, nonempty leaf set only after centered-parent and repeated-group classification do not apply.

- [ ] **Step 6: Integrate after body lattice and before legacy layout**

In `_ruled_multilevel_template_header_projection_for_template`, preserve `body_lattice_with_header_anchors` priority, then call the sparse analyzer, then retain existing page35-37 logic unchanged.

- [ ] **Step 7: Run geometry tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_r2_regression.RuledSparseTemplateGeometryTests
```

Expected: all geometry tests pass.

### Task 4: Project Template Bodies and Markdown

**Files:**
- Modify: `parsers/pdf/postprocess.py`
- Modify: `api/main.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Project body rows losslessly**

Map owned/page words below the leaf band and above notes to logical columns, retaining text, bbox, column, and source. Preserve ambiguous form rows outside the semantic grid.

- [ ] **Step 2: Emit semantic grids for all three sparse modes**

For visible parents, emit parent and leaf header rows. For blank group spans, emit the repeated leaf header without fabricated labels and retain blank spans in projection metadata.

- [ ] **Step 3: Extend exact Markdown consumption**

Feed only helper-validated `consumed_header_row_texts` and `consumed_body_row_texts` into structure-template bullet suppression. Do not add fuzzy matching.

- [ ] **Step 4: Run page 45/49/52 tests and verify GREEN**

Run the three Task 1 test methods. Expected: all pass.

### Task 5: Focused Regression and Inventory

**Files:**
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Run affected family tests**

Run geometry tests plus pages 35, 36, 37, 41, 45, 49, 52, 53, and 54. Expected: all pass.

- [ ] **Step 2: Compile and check diffs**

```powershell
.\.venv\Scripts\python.exe -m py_compile api/main.py parsers/pdf/postprocess.py tests/parser_tests/test_r2_regression.py
git diff --check -- api/main.py parsers/pdf/postprocess.py tests/parser_tests/test_r2_regression.py
```

Expected: exit code 0, allowing only line-ending warnings.

- [ ] **Step 3: Run full r2 projection inventory**

Parse `D:\AutoIND-Pro\r2.pdf`, list every ruled multilevel projection and mode, and confirm existing modes remain stable while only strongly evidenced new sparse templates are added.

- [ ] **Step 4: Decide whether to expand regression scope**

Run complete r2/eCTD/A-tst/benchmark suites only if focused tests, ownership checks, or inventory reveal unexpected impact.

### Task 6: Review and Mirror Sync

**Files:**
- Modify mirror: `D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai\parsers\pdf\postprocess.py`
- Modify mirror: `D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai\api\main.py`
- Modify mirror: `D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai\tests\parser_tests\test_r2_regression.py`

- [ ] **Step 1: Request independent code review**

Review ambiguity rejection, title/noise exclusion, evidence preservation, ownership, and regression scope. Fix Critical or Important findings before syncing.

- [ ] **Step 2: Sync accepted changes narrowly**

Copy files only when the mirror baseline matches. If `api/main.py` differs independently, apply only the sparse-template Markdown block.

- [ ] **Step 3: Verify mirror**

Compile the three files, run geometry plus page45/49/52 tests, compare exact hashes for safely copied files, and compare the relevant API function block for narrow patches.

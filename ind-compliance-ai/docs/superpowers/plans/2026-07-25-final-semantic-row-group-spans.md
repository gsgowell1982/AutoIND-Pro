# Final Semantic Row Group Spans Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate source-backed body rowspans from each table's final semantic grid so row grouping remains correct after PDF rows are expanded or repaired.

**Architecture:** Study-metric projection will preserve word-level lineage for every projected semantic row. A table-family-independent final resolver will infer group-key, static, and detail columns from the stable semantic grid, validate candidate groups against ordered provenance, invalidate stale body spans, and publish final semantic row groups before canonical span adaptation.

**Tech Stack:** Python 3, existing PDF postprocessing pipeline, PyMuPDF word geometry already supplied by the parser, `unittest`, existing HTML/Markdown table renderer.

**Execution constraint:** Work in the existing shared dirty `main` worktree. Do not create commits, reset files, clean the tree, or overwrite unrelated user changes. Use focused diffs and copy only task-owned files to the mirror after verification.

---

### Task 1: Lock the Final Semantic Grouping Contract

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`
- Reference: `docs/superpowers/specs/2026-07-25-final-semantic-row-group-spans-design.md`

- [ ] **Step 1: Add a unit test for variable group sizes and static columns**

Add `FinalSemanticRowGroupResolutionTests.test_resolves_groups_from_final_semantic_rows_and_provenance` with a synthetic grid containing group sizes 2 and 3. Supply ordered `semantic_row_provenance` records with distinct source y-ranges. Call the wished-for API:

```python
postprocess._resolve_final_semantic_row_groups([table])
postprocess._project_canonical_table_cell_spans([table])
```

Assert:

```python
self.assertEqual(
    [
        (group["start_semantic_row"], group["rowspan"], group["static_cols"], group["detail_cols"])
        for group in table["semantic_row_groups"]
    ],
    [(1, 2, [0, 1], [2, 3]), (3, 3, [0, 1], [2, 3])],
)
```

Assert canonical body spans exist for both static columns and use `coordinate_space == "semantic_grid"`.

- [ ] **Step 2: Add fail-closed unit tests**

Add separate tests that prove:

```python
# No provenance: no semantic_row_groups and no new body spans.
# A continuation row with no detail values: candidate group terminates or is rejected.
# A contradictory nonempty static-column continuation value: that column is not spanned.
# Two adjacent equal-text anchors with separate y regions remain separate groups.
```

- [ ] **Step 3: Add a stale-span invalidation test**

Build a table carrying an old `presentation_spans` body entry whose row count came from a physical grid. Mark the table as having replaced its semantic grid, invoke the resolver, and assert the stale body span is removed before final canonical projection.

- [ ] **Step 4: Run the new unit tests and verify RED**

Run:

```powershell
python -m unittest tests.parser_tests.test_r2_regression.FinalSemanticRowGroupResolutionTests -v
```

Expected: FAIL because `_resolve_final_semantic_row_groups` and the final grouping contract do not yet exist.

### Task 2: Preserve Word-Level Semantic Row Provenance

**Files:**
- Modify: `parsers/pdf/postprocess.py:40380`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Add a focused provenance test**

Construct positioned word rows containing one group anchor and multiple detail records. Exercise the study-metric row projection and assert each output record contains:

```python
{
    "cells": [...],
    "source_word_refs": [...],
    "source_y_min": ...,
    "source_y_max": ...,
    "projection_source": "study_metric_grouped_matrix_projection",
}
```

Verify packed source content split into multiple semantic records preserves strictly increasing y-order.

- [ ] **Step 2: Run the provenance test and verify RED**

Run the exact new test with `python -m unittest ... -v`.

Expected: FAIL because `_project_study_metric_data_rows()` currently returns text grids without lineage.

- [ ] **Step 3: Introduce a projected-row helper**

Add a focused helper near `_project_study_metric_data_rows`:

```python
def _project_study_metric_data_row(
    row: list[dict[str, Any]],
    anchors: list[float],
    *,
    expected_col_count: int,
) -> dict[str, Any] | None:
    ...
```

It returns normalized cell text plus deduplicated source word references and the row's y-range. Keep `_project_words_to_study_metric_columns()` as the text projection primitive; do not add table-family-specific grouping here.

- [ ] **Step 4: Carry provenance through projection payloads**

Update the AUC and impurity projection builders so their payloads contain both:

```python
"semantic_grid": [header, *projected_cells]
"semantic_row_provenance": [header_provenance, *projected_provenance]
```

Header provenance must declare its role and projection source. Body provenance must use final semantic row indices.

- [ ] **Step 5: Install provenance when replacing the semantic grid**

Update `_apply_study_metric_grouped_matrix_projection()` to set `table["semantic_row_provenance"]` and a coordinate-generation marker such as:

```python
table["semantic_grid_generation"] = "study_metric_grouped_matrix_projection"
table["semantic_grid_replaced"] = True
```

Before replacement, remove stale body `presentation_spans`, body `cell_spans`, and coordinate-dependent final groups while preserving header span evidence.

- [ ] **Step 6: Run the provenance tests and existing study-metric tests**

Run:

```powershell
python -m unittest tests.parser_tests.test_r2_regression.FinalSemanticRowGroupResolutionTests -v
python -m unittest tests.parser_tests.test_r2_regression.R2RegressionTests.test_page94_raw_material_table_reclaims_quality_and_batch_header_rows -v
```

Expected: provenance tests PASS; the page 94 test may still fail on the old rowspan assertion until Task 4.

### Task 3: Implement the Generic Final Row Group Resolver

**Files:**
- Modify: `parsers/pdf/postprocess.py:32984`
- Modify: `parsers/pdf/postprocess.py:13916`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Add provenance validation helpers**

Add small helpers that:

```python
def _semantic_row_source_interval(table, row_index) -> tuple[float, float] | None: ...
def _semantic_rows_have_strict_source_order(table, start_row, end_row) -> bool: ...
def _semantic_body_boundary_row(table, row_index) -> bool: ...
```

They must reject missing, reversed, overlapping-conflicting, or non-body lineage. Shared physical source rows are allowed only when the projected word references or sub-row y-order remain distinguishable.

- [ ] **Step 2: Infer candidate group-key columns**

Implement:

```python
def _final_semantic_group_key_candidates(
    semantic_grid: list[list[str]],
    *,
    header_row_count: int,
) -> list[int]:
    ...
```

Score columns by nonempty anchors separated by blank continuation runs and by their lower anchor frequency relative to row-level columns. Prefer earlier columns only as a tie-breaker. Do not inspect header text or value syntax.

- [ ] **Step 3: Resolve groups and column roles**

Implement:

```python
def _resolve_final_semantic_row_groups(tables: list[dict[str, Any]]) -> None:
    ...
```

For each candidate key column:

```python
start = nonempty key row
end = row before next nonempty key or body end
static_cols = anchor-nonempty columns blank throughout continuation
detail_cols = columns with row-level content in continuation rows
```

Accept a group only when it spans at least two rows, every continuation row has detail content, source order is valid, and no boundary row occurs inside it. Select the candidate column set with the strongest complete-group coverage and no overlap.

Emit `semantic_row_groups` with stable group IDs, source references, confidence, static columns, and detail columns.

- [ ] **Step 4: Project final groups into presentation spans**

Extend `_source_body_row_group_candidates()` or add a dedicated adapter so every static cell in an accepted final semantic group becomes a body presentation span. Use the final semantic row directly; do not remap by text search.

The resulting span carries:

```python
{
    "row": start_semantic_row,
    "col": static_col,
    "rowspan": rowspan,
    "colspan": 1,
    "text": semantic_grid[start_semantic_row][static_col],
    "span_group_id": group_id,
    "source": "final_semantic_row_group_resolution",
    "source_row_refs": [...],
}
```

- [ ] **Step 5: Install the resolver at the correct pipeline boundary**

Call `_resolve_final_semantic_row_groups(table_nodes)` after all table-family semantic projections and before `_project_source_body_row_groups_to_presentation_spans()` and `_project_canonical_table_cell_spans()`.

Do not rerun the old raw-grid detector at this stage.

- [ ] **Step 6: Run unit tests and verify GREEN**

Run:

```powershell
python -m unittest tests.parser_tests.test_r2_regression.FinalSemanticRowGroupResolutionTests -v
```

Expected: all final semantic row-group tests PASS.

### Task 4: Replace the Incorrect Page 94 Regression Baseline

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py:9560`

- [ ] **Step 1: Replace the partial rowspan assertion**

Remove the assertion requiring `<td rowspan="3">94NA103</td>` and assert the complete first-column span map:

```python
expected = {
    "LN125": 3,
    "94NA103": 5,
    "95NA215": 5,
    "95NB003": 2,
    "96NB101": 7,
}
```

Validate both AST `cell_spans` and rendered HTML.

- [ ] **Step 2: Assert complete group coverage**

For each expected first-column span, verify:

```python
span["row"] + span["rowspan"] == next_group_start
span["coordinate_space"] == "semantic_grid"
span["source_cell_refs"] or span["span_group_id"]
```

Assert that covered continuation rows do not emit extra empty first-column `<td>` cells.

- [ ] **Step 3: Assert static-column source fidelity**

For each accepted group, assert the source-backed static columns `批号`, `纯度(%)`, `A`, `B`, and `C` share the same group extent. Confirm `试验编号` and `试验类型` remain ordinary row-level cells.

- [ ] **Step 4: Run page 94 regression and verify GREEN**

Run:

```powershell
python -m unittest tests.parser_tests.test_r2_regression.R2RegressionTests.test_page94_raw_material_table_reclaims_quality_and_batch_header_rows -v
```

Expected: PASS with rowspans 3, 5, 5, 2, and 7.

### Task 5: Protect Existing Tables and Negative Controls

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`
- Modify only if necessary: `parsers/pdf/postprocess.py`

- [ ] **Step 1: Add synthetic negative controls**

Add tests proving no final group is emitted for:

```python
# sparse rows with no source lineage
# sparse rows separated by a note/subtotal boundary
# first-column blanks where later-column content is incomplete
# repeated values that have no blank continuation run
```

- [ ] **Step 2: Run existing body-span regression tests**

Run the focused tests covering source-backed and canonical body spans, including pages 85 and 102-105.

Expected: existing source-backed spans remain valid and no stale legacy span wins over final semantic evidence.

- [ ] **Step 3: Inspect changed-table span audit**

Parse r2 and print for every business table with final semantic groups:

```text
page, table_id, group_key_col, group sizes, static cols, detail cols, confidence
```

Manually inspect all newly affected tables. If the resolver affects unrelated table families without complete provenance, tighten generic evidence instead of adding exclusions.

### Task 6: Full Verification and Mirror Synchronization

**Files:**
- Authoritative files expected to change:
  - `parsers/pdf/postprocess.py`
  - `tests/parser_tests/test_r2_regression.py`
- Mirror equivalents under: `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai/`

- [ ] **Step 1: Run syntax and focused verification**

Run:

```powershell
python -m py_compile parsers/pdf/postprocess.py tests/parser_tests/test_r2_regression.py
python -m unittest tests.parser_tests.test_r2_regression.FinalSemanticRowGroupResolutionTests -v
python -m unittest tests.parser_tests.test_r2_regression.R2RegressionTests.test_page94_raw_material_table_reclaims_quality_and_batch_header_rows -v
```

- [ ] **Step 2: Run the full protected suites**

Run:

```powershell
python -m unittest tests.parser_tests.test_r2_regression -v
python -m unittest tests.deterministic_tests.test_parse_markdown_export -v
```

Expected: zero failures and zero errors.

- [ ] **Step 3: Run the all-table canonical span audit**

Parse r2 and verify every canonical span:

```python
row >= 0
col >= 0
row + rowspan <= active_grid_row_count
col + colspan <= active_grid_column_count
no overlap within the same role and coordinate space
body spans have provenance or a stable span_group_id
```

Render every spanned business table and verify no renderer exception occurs.

- [ ] **Step 4: Review the authoritative diff**

Run `git diff --check` and inspect only the task-owned hunks. Confirm no page-, section-, table-ID-, batch-, or expected-value condition entered production code.

- [ ] **Step 5: Synchronize only verified files to the mirror**

Compare authoritative and mirror versions before copying. Preserve unrelated mirror changes. Copy only the files changed by this task, then compare SHA-256 hashes.

- [ ] **Step 6: Run focused mirror verification**

Run syntax checks, final semantic row-group unit tests, and the page 94 regression from the mirror working directory.

Expected: authoritative and mirror behavior match with identical hashes for synchronized files.

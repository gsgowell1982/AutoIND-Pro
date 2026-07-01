# Table Semantic Projection v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first production-safe semantic table projection layer so table structure quality improves without violating IND/eCTD regression baselines.

**Architecture:** Keep existing table detectors and raw evidence intact. Add an additive post-AST semantic projection module that classifies recurring table families and emits logical structure metadata, then allow high-confidence projections to update public grids through a single interface.

**Tech Stack:** Python, current AutoIND PDF parser, unittest, PyMuPDF evidence already exposed by the parser.

---

### Task 1: Metric Report Fields

**Files:**
- Modify: `scripts/opendataloader_goal_metrics.py`
- Modify: `tests/deterministic_tests/test_opendataloader_goal_metrics.py`

- [ ] **Step 1: Write failing test**

Add a deterministic test that builds a tiny official evaluation payload containing `teds=0.5` and `teds_s=0.75`, runs `compute_goal_metrics`, and asserts:

```python
score = payload["metrics"]["goal_score"]
official = payload["metrics"]["official_reference"]
self.assertEqual(score["table_teds_mean"], 0.5)
self.assertEqual(score["table_teds_s_mean"], 0.75)
self.assertEqual(score["table_quality_mean"], 0.75)
self.assertEqual(official["teds_mean"], 0.5)
self.assertEqual(official["teds_s_mean"], 0.75)
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.deterministic_tests.test_opendataloader_goal_metrics -v
```

Expected: failure because `table_teds_mean` / `table_teds_s_mean` are missing.

- [ ] **Step 3: Implement**

Compute table-specific goal means from official per-document `teds` and `teds_s` values for documents whose ground truth has tables, and add them to `goal_score` plus `official_reference`.

- [ ] **Step 4: Verify GREEN**

Run the same deterministic test module and confirm it passes.

### Task 2: Semantic Projection Module

**Files:**
- Create: `parsers/pdf/table_modules/semantic_projection.py`
- Modify: `parsers/pdf/postprocess.py`
- Test: `tests/parser_tests/test_table_semantic_projection_v2.py`

- [ ] **Step 1: Write failing unit tests**

Create direct table-dict tests for:

```python
from parsers.pdf.table_modules.semantic_projection import apply_table_semantic_projection_v2
```

Cases:

- two-column inventory with repeated first row in the next fragment should expose `table_family="two_column_inventory"` and `semantic_grid` with list items kept inside the correct cell.
- comparison matrix with a blank top-left header and row-label column should expose `table_family="comparison_matrix"` and preserve the stub column in `semantic_header`.
- sparse grouped body rows should preserve existing `row_groups` and emit `logical_cells` with rowspan metadata.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_table_semantic_projection_v2 -v
```

Expected: import or assertion failure because the module does not exist.

- [ ] **Step 3: Implement minimal module**

Implement:

```python
def apply_table_semantic_projection_v2(table: dict[str, Any]) -> bool:
    ...
```

The function should:

- classify families from evidence shape;
- keep raw evidence unchanged;
- write `semantic_projection_v2`;
- write `semantic_grid`, `semantic_header`, and `logical_cells`;
- only update `display_grid/data_grid/grid` when the projection is high confidence and strictly improves duplicate/split row structure.

- [ ] **Step 4: Wire into parser postprocess**

Call `apply_table_semantic_projection_v2(table)` after existing header/rowspan projection and before vector OCR compaction in `parsers/pdf/postprocess.py`.

- [ ] **Step 5: Verify GREEN**

Run the new test module and table header grammar tests.

### Task 3: Benchmark Weak-Sample Regression

**Files:**
- Modify: `tests/parser_tests/test_opendataloader_benchmark_regression.py`

- [ ] **Step 1: Write failing tests**

Add assertions:

- `01030000000121` should be represented as one semantic two-column inventory table, not two independent semantic tables split by a repeated row.
- `01030000000120` should not emit the tail one-column fragment `in long rods` as an independent semantic table; the flowchart matrix should retain connector text/metadata.
- `01030000000182` should have a semantic comparison matrix with the projected row-header/stub role retained.

- [ ] **Step 2: Verify RED**

Run targeted tests and confirm failures match current AST behavior.

- [ ] **Step 3: Implement merge/projection refinements**

Extend the semantic projection layer or postprocess call site to merge adjacent semantic fragments only when:

- same page and compatible source family;
- boundary row is repeated or semantically continued;
- no independent caption/title between them;
- merged grid improves row/column completeness;
- body prose, notes, and captions are not absorbed.

- [ ] **Step 4: Verify GREEN**

Run targeted OpenDataLoader tests.

### Task 4: Protected Gates and State

**Files:**
- Sync changed runtime/test files to mirror after primary verification.
- Update state files only after verification.

- [ ] **Step 1: Primary gates**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:IND_FORMULA_OCR_ENABLED=''
$env:IND_A_TST_REGRESSION_PDF='D:/AutoIND-Pro/A-tst.pdf'
.\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_opendataloader_benchmark_regression tests.parser_tests.test_table_header_grammar tests.deterministic_tests.test_opendataloader_goal_metrics -v
.\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_a_tst_regression -v
.\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_two_column_literature_regression -v
.\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_r2_regression -v
.\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_ectd_regression_sample -v
```

- [ ] **Step 2: Mirror sync and mirror gate**

Copy only changed files to:

```text
D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai
```

Then rerun the practical mirror gate with the primary venv if the mirror has no local venv.

- [ ] **Step 3: Optional metric rerun**

If targeted and protected tests are green, run a 200-sample benchmark only if time allows; otherwise run a focused changed-sample benchmark and record that the full score rerun is pending.

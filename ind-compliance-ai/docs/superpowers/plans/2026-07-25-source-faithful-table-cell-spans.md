# Source-Faithful Table Cell Spans Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make table rowspans and colspans source-faithful and uniform in the AST and HTML for real IND tables, including r2.pdf sections 2.6.7.8A, 2.6.7.8B, and 2.6.7.9A.

**Architecture:** Add canonical `cell_spans` records after semantic table projection, adapting existing high-confidence header and body spans without mutating any raw/display/semantic grid. Resolve body groups through source-row lineage, merge physical fragments into logical-chain spans in the API, and make semantic HTML prefer canonical spans while retaining legacy fallback for tables not yet migrated.

**Tech Stack:** Python 3, existing PDF postprocessor, API Markdown/HTML exporter, `unittest`, project r2 regression fixtures.

---

## File Map

- Modify `parsers/pdf/postprocess.py`: canonical span projection, span validation/provenance, genotoxicity header classification, and source-lineage body span mapping.
- Modify `api/main.py`: canonical span consumption, logical continuation-chain span merging, and semantic HTML header/span rendering.
- Modify `tests/parser_tests/test_r2_regression.py`: parser/AST regression coverage for pages 102-105 and repeated-label lineage behavior.
- Modify `tests/deterministic_tests/test_parse_markdown_export.py`: deterministic continuation-chain and HTML rendering contracts.
- Create `docs/superpowers/plans/2026-07-25-source-faithful-table-cell-spans.md`: this execution plan.

### Task 1: Canonical AST Span Adapter

**Files:**
- Modify: `parsers/pdf/postprocess.py:13917`
- Modify: `parsers/pdf/postprocess.py:32973`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Write the failing canonical-contract test**

Add a focused synthetic test that invokes `_project_canonical_table_cell_spans([table])` and asserts normalized header/body records:

```python
self.assertEqual(
    [(s["role"], s["row"], s["col"], s["rowspan"], s["colspan"])
     for s in table["cell_spans"]],
    [("header", 0, 0, 2, 1), ("body", 2, 0, 3, 1)],
)
self.assertTrue(all(s["coordinate_space"] == "semantic_grid" for s in table["cell_spans"]))
self.assertTrue(all(s["source_table_ids"] == ["tbl_test"] for s in table["cell_spans"]))
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest tests.parser_tests.test_r2_regression.CanonicalTableCellSpanTests -v`

Expected: FAIL because `_project_canonical_table_cell_spans` or `cell_spans` does not exist.

- [ ] **Step 3: Implement the canonical adapter and invariants**

Add `_project_canonical_table_cell_spans`, `_canonical_table_cell_span`, and overlap validation. Normalize `span_header_cells` as `role="header"` and `presentation_spans` as `role="body"`, assign stable IDs, preserve source pages/table IDs/cell refs/evidence/confidence, reject unit spans and conflicting overlaps, and store an audit under `semantic_projection_v2.canonical_cell_span_projection`.

- [ ] **Step 4: Insert canonical projection after legacy span producers**

Call:

```python
_project_source_body_row_groups_to_presentation_spans(table_nodes)
_project_canonical_table_cell_spans(table_nodes)
```

This ordering makes legacy fields compatibility inputs, while `cell_spans` becomes the final parser contract.

- [ ] **Step 5: Run the canonical-contract test and verify GREEN**

Run: `python -m unittest tests.parser_tests.test_r2_regression.CanonicalTableCellSpanTests -v`

Expected: PASS with deterministic span ordering and provenance.

### Task 2: Classify True Multilevel Headers Versus Multiline Cells

**Files:**
- Modify: `parsers/pdf/postprocess.py:32814`
- Modify: `parsers/pdf/postprocess.py:33420`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Add failing header classification tests**

Extend the pages 102-104 regression to assert:

```python
self.assertEqual(
    {(s["row"], s["col"], s["rowspan"], s["colspan"])
     for s in page102["cell_spans"] if s["role"] == "header"},
    {(0, 0, 2, 1), (0, 1, 2, 1), (0, 2, 2, 1), (0, 3, 1, 5)},
)
self.assertFalse(any(s["role"] == "header" for s in page103["cell_spans"]))
self.assertFalse(any(s["role"] == "header" for s in page104["cell_spans"]))
```

Also assert the seven 2.6.7.8B headers and five 2.6.7.9A headers remain independent semantic cells.

- [ ] **Step 2: Run the focused page test and verify RED**

Run: `python -m unittest tests.parser_tests.test_r2_regression.R2RegressionTests.test_pages102_to_104_genotoxicity_result_matrices_stitch_cross_page_continuations -v`

Expected: FAIL because 8A lacks two standalone header rowspans and 8B contains a false four-column colspan.

- [ ] **Step 3: Complete 8A standalone header rowspans**

Change `_genotoxicity_bacterial_reverse_mutation_header_schema` so its span list starts with three two-row standalone anchors before the true five-column experiment parent:

```python
spans = [
    {"row": 0, "col": 0, "rowspan": 2, "colspan": 1, "text": header[0], "source": source},
    {"row": 0, "col": 1, "rowspan": 2, "colspan": 1, "text": header[1], "source": source},
    {"row": 0, "col": 2, "rowspan": 2, "colspan": 1, "text": dose_header, "source": source},
]
```

- [ ] **Step 4: Reject multiline-leaf false parent groups**

In `_genotoxicity_chromosomal_aberration_header_schema`, reject a candidate parent when its normalized text is equal to or a prefix of the first child in the same start column. Treat the parent and unit fragments as one wrapped leaf header; require independent geometry/children for a true colspan.

- [ ] **Step 5: Re-run the focused page test and verify GREEN**

Run the command from Step 2.

Expected: PASS with exact AST header spans for 8A and no invented header spans for 8B/9A.

### Task 3: Resolve Repeated Body Groups by Source Row Lineage

**Files:**
- Modify: `parsers/pdf/postprocess.py:32973`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Write a failing repeated-label test**

Create a synthetic semantic grid containing two `MM-180801` groups in the same column, with distinct `start_data_row`/`end_data_row` and semantic row provenance. Assert both become body spans rather than being rejected as text-ambiguous.

```python
self.assertEqual(
    [(s["row"], s["col"], s["rowspan"], s["text"])
     for s in table["presentation_spans"]],
    [(2, 1, 3, "MM-180801"), (7, 1, 2, "MM-180801")],
)
```

- [ ] **Step 2: Run the repeated-label test and verify RED**

Run: `python -m unittest tests.parser_tests.test_r2_regression.SourceBodyRowGroupPresentationTests -v`

Expected: FAIL because text uniqueness currently rejects repeated labels.

- [ ] **Step 3: Add lineage-first group positioning**

Pass the whole table to `_source_body_row_group_presentation_span`. Map `start_data_row`/`end_data_row` through `semantic_row_provenance` source-row references, verify column/text/contiguous extent, and use text-window uniqueness only as a compatibility fallback when lineage is absent.

- [ ] **Step 4: Preserve physical fragment identity**

Include source row references in accepted presentation spans and derive a stable `span_group_id` from table/column/source group coordinates. This gives continuation merging a lineage key without relying on label uniqueness.

- [ ] **Step 5: Run lineage and page 102-104 parser tests**

Run:

```powershell
python -m unittest tests.parser_tests.test_r2_regression.SourceBodyRowGroupPresentationTests -v
python -m unittest tests.parser_tests.test_r2_regression.R2RegressionTests.test_pages102_to_104_genotoxicity_result_matrices_stitch_cross_page_continuations -v
```

Expected: PASS; both repeated `MM-180801` groups are retained with source-backed spans.

### Task 4: Merge Canonical Spans Across Continued Table Chains

**Files:**
- Modify: `api/main.py:3298`
- Test: `tests/deterministic_tests/test_parse_markdown_export.py`

- [ ] **Step 1: Write a failing logical-chain span test**

Build a root plus continuation where a terminal body span continues through inherited omitted-prefix rows. Assert the merged table has one global canonical span:

```python
self.assertEqual(
    [(s["coordinate_space"], s["row"], s["col"], s["rowspan"])
     for s in merged["cell_spans"] if s["role"] == "body"],
    [("logical_table_chain", 2, 1, 5)],
)
self.assertEqual(span["source_table_ids"], ["tbl_root", "tbl_cont"])
self.assertEqual(span["source_pages"], [102, 103])
```

- [ ] **Step 2: Run the deterministic chain test and verify RED**

Run: `python -m unittest tests.deterministic_tests.test_parse_markdown_export.ParseMarkdownExportTests.test_continued_table_chain_extends_canonical_body_span -v`

Expected: FAIL because `_merge_continued_table_chain` only extends legacy presentation spans.

- [ ] **Step 3: Merge canonical physical fragments**

Update `_merge_continued_table_chain` to translate physical row offsets into logical-chain coordinates, extend matching terminal/leading fragments by `span_group_id` or verified inherited-prefix continuity, and union source pages/table IDs/cell refs. Keep the existing legacy merge path for tables without `cell_spans`.

- [ ] **Step 4: Validate merged spans**

Reject out-of-grid/overlapping logical spans, sort by `(row, col, role)`, and expose `coordinate_space="logical_table_chain"` on the merged owner without modifying physical table AST nodes.

- [ ] **Step 5: Run deterministic continuation tests**

Run: `python -m unittest tests.deterministic_tests.test_parse_markdown_export.ParseMarkdownExportTests -v`

Expected: PASS for canonical and legacy continuation behavior.

### Task 5: Make Semantic HTML Consume Canonical Spans First

**Files:**
- Modify: `api/main.py:2369`
- Modify: `api/main.py:2859`
- Test: `tests/deterministic_tests/test_parse_markdown_export.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Add failing canonical-rendering tests**

Assert a table with `cell_spans` renders each anchor once, suppresses covered cells, and ignores contradictory legacy spans. For the r2 export assert exact structural fragments:

```python
self.assertIn('<th rowspan="2">代谢活化</th>', markdown)
self.assertIn('<th colspan="5">实验#1 回复突变体菌落计数(平均值±SD)</th>', markdown)
self.assertNotIn('<th colspan="4">细胞毒性', markdown)
self.assertIn('<td rowspan="5">MM-180801</td>', markdown)
```

- [ ] **Step 2: Run rendering tests and verify RED**

Run:

```powershell
python -m unittest tests.deterministic_tests.test_parse_markdown_export.ParseMarkdownExportTests -v
python -m unittest tests.parser_tests.test_r2_regression.R2RegressionTests.test_page102_to_104_genotoxicity_markdown_preserves_source_cell_spans -v
```

Expected: FAIL because rendering still selects legacy span surfaces and repeats/misgroups headers.

- [ ] **Step 3: Prefer canonical spans in renderer helpers**

Make `_semantic_table_span_cells` return validated `cell_spans` when the key is present, with legacy `span_header_cells`/`presentation_spans` fallback only when canonical data is absent. Update `_semantic_html_table_header_row_count` to derive header depth from canonical header spans.

- [ ] **Step 4: Disable blank-cell inference under canonical contract**

When `cell_spans` exists, do not invent colspans or rowspans from blank neighboring cells. Preserve line wrapping inside a single `<th>` and render true anchors using explicit `rowspan`/`colspan` attributes.

- [ ] **Step 5: Run deterministic and r2 rendering tests**

Run the commands from Step 2.

Expected: PASS with source-faithful 8A/8B/9A structures and no duplicated covered cells.

### Task 6: Regression Verification and Mirror Synchronization

**Files:**
- Verify: `parsers/pdf/postprocess.py`
- Verify: `api/main.py`
- Verify: `tests/parser_tests/test_r2_regression.py`
- Verify: `tests/deterministic_tests/test_parse_markdown_export.py`
- Sync to: `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai`

- [ ] **Step 1: Compile modified Python modules**

Run: `python -m py_compile parsers/pdf/postprocess.py api/main.py tests/parser_tests/test_r2_regression.py tests/deterministic_tests/test_parse_markdown_export.py`

Expected: exit code 0.

- [ ] **Step 2: Run focused table-span regression suites**

Run:

```powershell
python -m unittest tests.deterministic_tests.test_parse_markdown_export.ParseMarkdownExportTests -v
python -m unittest tests.parser_tests.test_r2_regression.SourceBodyRowGroupPresentationTests -v
python -m unittest tests.parser_tests.test_r2_regression.CanonicalTableCellSpanTests -v
```

Expected: all PASS.

- [ ] **Step 3: Run the protected r2 regression module**

Run: `python -m unittest tests.parser_tests.test_r2_regression -v`

Expected: all tests PASS; if runtime is long, keep the command alive and report its final count rather than treating silence as completion.

- [ ] **Step 4: Inspect the actual pages 102-105 AST and export**

Use the existing project parse/export fixture to verify `cell_spans`, header arrays, logical-chain spans, and exact HTML. Confirm all source rows, positive controls, notes, and continuation rows remain present.

- [ ] **Step 5: Review only task-owned diffs**

Run:

```powershell
git diff -- parsers/pdf/postprocess.py api/main.py tests/parser_tests/test_r2_regression.py tests/deterministic_tests/test_parse_markdown_export.py docs/superpowers/plans/2026-07-25-source-faithful-table-cell-spans.md
git status --short
```

Expected: intended span changes are visible; unrelated dirty-worktree changes remain untouched.

- [ ] **Step 6: Synchronize authoritative files to the mirror**

Copy only the modified authoritative files to the identical relative paths under `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai`, then compare SHA-256 hashes for every copied file.

- [ ] **Step 7: Run a mirror smoke verification**

Run the focused canonical/deterministic tests from the mirror path and verify the same AST/render outcome.

- [ ] **Step 8: Preserve the existing dirty-worktree baseline**

Do not broad-stage or commit production files that contain user/previous-session changes. Record the tested file list, test commands, hash equality, and remaining unrelated status in the handoff.

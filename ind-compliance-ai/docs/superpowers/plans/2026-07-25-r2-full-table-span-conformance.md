# R2 Full Table Span Conformance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every source-backed rowspan and colspan in r2.pdf is represented by canonical `cell_spans` and rendered uniformly as semantic HTML, without reviving stale or conflicting legacy span hints.

**Architecture:** Keep canonical `cell_spans` as the only authoritative rendering contract. Resolve ambiguous source row numbering by matching both plausible source-row coordinate conventions against semantic-row lineage and cell content, materialize any canonical multilevel header before rendering, and require evidence Markdown to use semantic HTML whenever the canonical contract contains a real span. Add a document-wide conformance test that validates coordinates, overlap, projected spans, and emitted HTML attributes for every r2 business table.

**Tech Stack:** Python 3, existing PDF postprocessor, API Markdown/HTML exporter, `unittest`, r2.pdf regression fixture.

---

### Task 1: Audit the Existing R2 Span Surface

**Files:**
- Test: `tests/parser_tests/test_r2_regression.py`

- [x] **Step 1: Add a full-document AST invariant test**

Assert that every canonical span has positive dimensions, stays within its coordinate grid, does not overlap another span of the same role, and retains source page/table/evidence provenance.

- [x] **Step 2: Add a full-document projected-render invariant test**

For each business table with non-empty canonical spans, build the evidence grid, project canonical coordinates, render semantic HTML, and assert that every projected canonical anchor emits its exact `rowspan` and/or `colspan` attribute.

- [x] **Step 3: Run the new tests and verify RED**

Run: `python -m unittest tests.parser_tests.test_r2_regression.R2RegressionTests.test_all_canonical_table_spans_are_structurally_valid tests.parser_tests.test_r2_regression.R2RegressionTests.test_all_canonical_table_spans_survive_evidence_rendering -v`

Expected: the structural test passes, while rendering fails for header-only canonical tables that still fall back to pipe Markdown or flattened header surfaces.

### Task 2: Recover Source-Backed Body Spans Across Row Numbering Conventions

**Files:**
- Modify: `parsers/pdf/postprocess.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [x] **Step 1: Add a failing lineage test for a one-based source row group**

Create a semantic grid whose provenance maps `display_row:4` to semantic row 2 and whose source group starts at row 4. Assert that `MM-180801` becomes a canonical body span at `(2, 0)` with `rowspan=8`.

- [x] **Step 2: Verify RED**

Run: `python -m unittest tests.parser_tests.test_r2_regression.SourceBackedBodySpanProjectionTests.test_resolves_one_based_source_group_by_exact_lineage_and_content -v`

Expected: FAIL because the current resolver assumes only the zero-based `start_data_row + 1` convention.

- [x] **Step 3: Implement evidence-ranked lineage candidates**

Build exact `display_row -> semantic_row` mappings from provenance. Evaluate both `start_data_row` and `start_data_row + 1`, retain only candidates whose anchor and full span window are compatible with the source group, and accept only a unique valid semantic start. Preserve the existing stable-offset fallback when exact references are sparse.

- [x] **Step 4: Verify GREEN and the actual page 105 table**

Assert `tbl_recovered_105_02` exposes body span `(2, 0, 8, 1, "MM-180801")` with page/table/source-row provenance.

### Task 3: Make Canonical Header Rendering Generic

**Files:**
- Modify: `api/main.py`
- Test: `tests/deterministic_tests/test_parse_markdown_export.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [x] **Step 1: Add failing deterministic tests**

Cover a flattened dose/sex header and a three-level metric/species/sex header. Assert that canonical header spans materialize the required header depth, preserve leaf headers, and render exact HTML spans.

- [x] **Step 2: Verify RED**

Run the new deterministic tests and confirm existing family-specific projection either flattens the span or bypasses canonical rendering.

- [x] **Step 3: Prefer the canonical header surface for every canonical header contract**

Call `_project_markdown_canonical_header_grid` for any table with canonical header spans, not only genotoxicity tables. Keep family-specific projectors as compatibility fallbacks only when canonical spans are absent.

- [x] **Step 4: Route every non-empty canonical span contract through evidence semantic HTML**

Update `_markdown_table_requires_evidence_semantic_html` so a non-empty canonical span list is sufficient. Continue to render ordinary tables without spans through their existing path.

- [x] **Step 5: Verify GREEN**

Run deterministic rendering tests and the full-document projected-render invariant.

### Task 4: Regression Verification and Mirror Synchronization

**Files:**
- Verify: `parsers/pdf/postprocess.py`
- Verify: `api/main.py`
- Verify: `tests/parser_tests/test_r2_regression.py`
- Verify: `tests/deterministic_tests/test_parse_markdown_export.py`
- Sync to: `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai`

- [x] **Step 1: Compile modified Python files**

Run `python -m py_compile` for all four modified modules.

- [x] **Step 2: Run focused deterministic and r2 span tests**

Verify the lineage resolver, canonical header materialization, page 105 body span, and full-document conformance tests.

- [x] **Step 3: Run complete protected suites**

Run all deterministic Markdown export tests and the complete r2 regression module.

- [x] **Step 4: Re-run the independent r2 inventory**

Expected: 56 business tables audited; no invalid canonical spans; no source-backed accepted span missing from canonical AST; every projected canonical span emits its exact HTML attribute.

- [x] **Step 5: Sync authoritative files and compare SHA-256**

Copy only the four modified code/test files and this plan to the mirror, compare hashes, then run mirror smoke tests.

- [x] **Step 6: Preserve the dirty worktree baseline**

Do not broad-stage, commit, reset, or clean production files containing prior work.

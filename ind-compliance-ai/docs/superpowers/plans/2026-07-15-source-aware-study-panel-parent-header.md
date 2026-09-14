# Source-Aware Study-Panel Parent Header Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover geometry-backed parent headers between populated study metadata and borderless tables while preserving source-aware metadata rows.

**Architecture:** Classify terminal panel rows before metadata continuation merging, retain one-to-many row provenance, and reuse the existing `pre_table_parent_header_projection` contract for ownership transfer and Markdown rendering.

**Tech Stack:** Python, PyMuPDF word geometry, existing PDF postprocessor, `unittest`.

---

### Task 1: Add Regression Contracts

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Add a synthetic test that releases a terminal parent header only when it spans multiple word-backed leaf columns.
- [ ] Add a synthetic test proving a merged metadata continuation retains both source block IDs.
- [ ] Run both synthetic tests and the real page-81 span test; confirm RED failures caused by the missing source-aware behavior.

### Task 2: Implement Source-Aware Ownership

**Files:**
- Modify: `parsers/pdf/postprocess.py`

- [ ] Add source-aware metadata row records with `text`, `bbox`, and `source_block_ids`.
- [ ] Add terminal parent-header partitioning based on table adjacency and leaf-column anchors.
- [ ] Pass page words into the table-adjacent study-panel builder.
- [ ] Persist `row_sources` on templates and synchronize them to page AST/content evidence.
- [ ] Update template parent-header candidate and release logic to consume source-aware records.

### Task 3: Verify Focused Surfaces

**Files:**
- Verify: `parsers/pdf/postprocess.py`
- Verify: `api/main.py`
- Verify: `tests/parser_tests/test_r2_regression.py`

- [ ] Run synthetic ownership tests.
- [ ] Run real page 81-83 parent-header regressions and focused Markdown projection tests.
- [ ] Run `py_compile` and `git diff --check` for touched files.
- [ ] Sync touched files to the mirror, verify hashes, and rerun the focused contracts there.

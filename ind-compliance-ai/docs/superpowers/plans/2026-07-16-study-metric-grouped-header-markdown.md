# Study Metric Grouped Header Markdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render source-backed grouped study-metric headers in IND-review Markdown without changing canonical semantic data or unrelated tables.

**Architecture:** Add a strict profile adapter in `api/main.py` and a reusable grouped-header row materializer driven by `header_column_groups`. Preserve parser AST contracts and fail closed when span evidence is absent or invalid.

**Tech Stack:** Python, `unittest`, existing IND-review Markdown projection pipeline.

---

### Task 1: Lock the presentation contract

**Files:**
- Modify: `tests/deterministic_tests/test_parse_markdown_export.py`

- [x] Add a fixture with AUC spanning four columns, mouse and rat spanning two columns each, and terminal single-column species headers.
- [x] Assert the complete parent/group/leaf row sequence in generated Markdown.
- [x] Add a no-span fixture and assert its semantic rows remain unchanged.
- [x] Run both tests and confirm the grouped fixture fails before production changes while the no-span guard passes.

### Task 2: Materialize authoritative group rows

**Files:**
- Modify: `api/main.py`

- [x] Add a `study_metric_grouped_matrix` profile adapter before generic semantic-grid fallback.
- [x] Add a grouped-header materializer that validates group coordinates, rejects overlaps, orders levels, and preserves leaf headers.
- [x] Avoid the generic repeated-cell heuristic because repeated `M/F` leaves can make the first data row look like an existing header row.
- [x] Run the deterministic tests and confirm both pass.

### Task 3: Protect real r2 behavior

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [x] Assert the page-92 AUC, species, and sex header rows in final IND-review Markdown.
- [x] Assert page 94 renders `特定杂质a` over A/B/C through the same generic contract.
- [x] Run focused page 85, 91, 92, and 94 regressions.
- [x] Run syntax checks, mirror synchronization, and mirror verification.

# Cross-Page Open Note Group Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `附加信息：` out of the page 86 table grid and render its page 87 continuation with the page 86 owner before the next section.

**Architecture:** Convert result-matrix note-label rows into table-owned note segments, preserve explicit cross-page ownership as authoritative, and reject duplicate same-panel ownership on the following table. Keep label and continuation as separate ordered segments.

**Tech Stack:** Python, parser semantic AST, Markdown projection, unittest regression tests.

---

### Task 1: Establish failing page 86-to-87 behavior

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Assert `附加信息：` is absent from page 86 `semantic_grid` and present as a table note.
- [ ] Assert the continuation is owned only by page 86 and absent from page 87 note blocks.
- [ ] Assert Markdown renders both note lines after page 86 CTD position and before `2.6.5.14`, exactly once.
- [ ] Run the focused tests and confirm failures match the current incorrect ownership and row placement.

### Task 2: Separate local note labels from the matrix grid

**Files:**
- Modify: `parsers/pdf/postprocess.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Stop appending detected note rows to the semantic grid.
- [ ] Attach detected note rows to the table as ordered below-table note segments.
- [ ] Preserve the projection's `note_rows` audit metadata.

### Task 3: Enforce authoritative cross-page ownership

**Files:**
- Modify: `parsers/pdf/postprocess.py`
- Modify: `api/main.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Exclude previous-table continuation nodes from the following study panel's top-note candidates.
- [ ] Make Markdown prefer explicit cross-page owner metadata over local marker scoring.
- [ ] Preserve local-label then continuation rendering order.

### Task 4: Focused verification and synchronization

**Files:**
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Run page 85-to-87 ownership and Markdown regressions.
- [ ] Run Python compilation and `git diff --check`.
- [ ] Synchronize the changed files and documents to the project mirror and compare SHA-256 hashes.

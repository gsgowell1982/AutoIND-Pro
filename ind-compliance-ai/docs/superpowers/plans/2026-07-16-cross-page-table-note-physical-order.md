# Cross-Page Table Note Physical Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve physical order and line boundaries for table notes spanning local and following pages.

**Architecture:** Replace source-whitelist physical ranking with a canonical physical-page resolver. Keep line rendering narrow by activating it automatically only when all meaningful segments have geometry and span multiple pages.

**Tech Stack:** Python, `unittest`, IND-review Markdown projection.

---

### Task 1: Lock the failure

**Files:**
- Modify: `tests/deterministic_tests/test_parse_markdown_export.py`

- [x] Add local legend, local statistical note, and following-page definition note segments.
- [x] Assert exact independent lines and physical order.
- [x] Run the test and confirm the old sorter fails.

### Task 2: Generalize physical ordering

**Files:**
- Modify: `api/main.py`

- [x] Resolve physical pages from the established occurrence fields.
- [x] Apply page/bbox ordering to every geometrically anchored table note, independent of source.
- [x] Render complete multi-page note chains as lines while preserving single-page behavior.
- [x] Run deterministic note-order and explicit note-group tests.

### Task 3: Protect r2 and mirror behavior

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [x] Assert page-97 notes appear as three lines in physical order.
- [x] Run focused cross-page note regressions and the deterministic Markdown module.
- [x] Run syntax checks, mirror synchronization, and mirror verification.

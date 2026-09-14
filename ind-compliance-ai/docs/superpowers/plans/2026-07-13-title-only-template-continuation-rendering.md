# Title-Only Template Continuation Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve a source-visible page-bottom continuation heading when its template body begins on the next page.

**Architecture:** Preserve title-source ownership in every structure-template AST node projection. Add one renderer eligibility predicate that distinguishes an evidence-backed title-only continuation anchor from an ordinary empty template, then use it at the existing early-return boundary so downstream heading visibility and rendering remain unchanged.

**Tech Stack:** Python, `unittest`, existing PDF AST and IND-review Markdown renderer.

---

### Task 1: Add Regression Coverage

**Files:**
- Modify: `parsers/pdf/postprocess.py`
- Modify: `tests/parser_tests/test_r2_regression.py`

- [x] Add a focused renderer test with an empty ordinary template and an empty `pending_body_on_next_page` continuation template.
- [x] Require the page-71 AST continuation node to preserve `title_source_block_id`.
- [x] Extend the page-71 regression to assert that `#### 2.6.7.14 (1)生殖毒性 试验编号(续)` occurs after the terminal `b-` note and before page-72 `日剂量(mg/kg) 0(对照)`.
- [x] Run the focused tests and confirm failure because the title-only template is currently returned before heading rendering.

### Task 2: Implement Renderability Contract

**Files:**
- Modify: `api/main.py`
- Modify: `parsers/pdf/postprocess.py`

- [x] Add a narrowly scoped helper that returns true for existing template content or an evidence-backed pending continuation title anchor.
- [x] Project `title_source_block_id` through initial, synchronized, and late-created structure-template AST nodes.
- [x] Replace the raw empty-content early return with the helper.
- [x] Run the focused tests and confirm they pass without rendering the ordinary empty template.

### Task 3: Verify Affected Scope

**Files:**
- Verify: `api/main.py`
- Verify: `tests/parser_tests/test_r2_regression.py`

- [x] Run the affected page-71 and nearby continuation-template tests.
- [x] Run `py_compile` on changed Python files.
- [x] Run `git diff --check` and inspect the scoped diff.
- [x] Do not commit, merge, push, or clean the user's existing worktree.

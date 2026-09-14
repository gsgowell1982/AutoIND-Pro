# Source-Aware Result-Matrix Trailing Colspan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover and render evidence-backed trailing colspan rows in grouped study result matrices.

**Architecture:** Merge table atoms with table-scoped page words, cluster trailing physical rows, project centered values against verified leaf centers, and reject incomplete semantic bindings.

**Tech Stack:** Python, unittest, existing PDF word geometry and semantic-grid projection helpers.

---

### Task 1: Establish RED behavior

- [ ] Replace the current `StopIteration` expectation path with an explicit expected trailing-label assertion.
- [ ] Add Markdown ordering assertions for study number, CTD location, and additional information.
- [ ] Run the real page-86 test and confirm it fails with an empty trailing-row projection.

### Task 2: Build the source-aware row adapter

- [ ] Extract a reusable table-atoms plus page-words merge helper.
- [ ] Restrict page-word fallback to the current table bbox.
- [ ] Cluster candidate rows and separate label/value atoms using leaf geometry.

### Task 3: Project spans and protect commits

- [ ] Feed source-aware candidate rows into centered colspan projection.
- [ ] Merge adjacent CTD value fragments before assigning group spans.
- [ ] Compare projected labels with expected display-grid labels before committing the semantic binding.
- [ ] Emit diagnostics and reject incomplete bindings.

### Task 4: Verify and synchronize

- [ ] Run page-86 trailing-span and Markdown tests.
- [ ] Run page-85/86 ownership, prelude, grouped-matrix, page-81 shared contracts, and deterministic Markdown tests as impact requires.
- [ ] Run syntax and diff checks, sync touched files to the mirror, verify hashes, and repeat focused mirror tests.
- [ ] Record the architecture decision and verification result in session state.

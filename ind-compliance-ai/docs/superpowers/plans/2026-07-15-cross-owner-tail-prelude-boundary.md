# Cross-Owner Tail and Prelude Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve a study prelude between a preceding table's cross-page notes and the next study title.

**Architecture:** Select the formal study title independently from generic hard boundaries, classify all preceding rows by destination owner, and recover missing prelude geometry from page words before committing the crop.

**Tech Stack:** Python, unittest, existing PDF word evidence and study-panel ownership helpers.

---

### Task 1: Establish RED contracts

- [ ] Add a real r2 page-86 test for template/AST prelude ownership, bbox, transfer plan, and Markdown order.
- [ ] Run the test and confirm it fails because `prelude_blocks` is absent.

### Task 2: Separate prelude and formal-title boundaries

- [ ] Make the study transfer boundary skip a prelude only when a later adjacent row is a valid study title.
- [ ] Keep generic result-matrix hard-boundary behavior for grids without a study title.

### Task 3: Recover prelude geometry

- [ ] Pass page words into late study-panel refinement.
- [ ] Merge page words into the visual row evidence adapter when table word evidence lacks a matching row.
- [ ] Verify the transfer plan assigns the prelude to the destination template before cropping.

### Task 4: Verify and synchronize

- [ ] Run direct page-86 RED/GREEN tests, page-85 prelude tests, page-85-to-86 note tests, and shared parent-header contracts.
- [ ] Run `py_compile`, `git diff --check`, mirror hash verification, and mirror focused tests.
- [ ] Record the architecture decision and verification result in session state.

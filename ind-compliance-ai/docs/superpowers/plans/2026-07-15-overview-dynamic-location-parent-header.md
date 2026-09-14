# Overview Dynamic Location Parent Header Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize overview `位置 -> 卷/页码` parent headers to nonclinical tables and preserve them across continuation pages.

**Architecture:** Detect parent spans from dynamic adjacent leaf columns plus source geometry. Store canonical leaf columns separately from presentation rows and inherit both leaves and group metadata across continuation links.

**Tech Stack:** Python, PDF word geometry, semantic table AST, Markdown projection, unittest.

---

### Task 1: Establish page 88-89 failures

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Assert page 88 has a `位置` group over leaf indexes 8-9.
- [ ] Assert page 88 semantic rows contain parent and child header rows before data.
- [ ] Assert page 89 inherits the same leaf schema and group metadata.
- [ ] Run focused tests and confirm failure on missing groups.

### Task 2: Detect dynamic location groups

**Files:**
- Modify: `parsers/pdf/postprocess.py`

- [ ] Find the adjacent `卷` plus `页码/部分` leaf pair dynamically.
- [ ] Pass its indexes to the existing geometry-based parent-word selector.
- [ ] Remove the PK-only and fixed-index restrictions.
- [ ] Keep the source-parent requirement as the negative guard.

### Task 3: Inherit canonical continuation schema

**Files:**
- Modify: `parsers/pdf/postprocess.py`

- [ ] Store `leaf_columns` in the overview projection.
- [ ] Read inherited leaves and groups from the parent projection.
- [ ] Build continuation header rows from inherited leaves and groups.
- [ ] Preserve local detection when a continuation repeats its header.

### Task 4: Focused verification and synchronization

**Files:**
- Test: `tests/parser_tests/test_r2_regression.py`
- Modify: `D:/ind-session/ENGINEERING_DECISIONS.md`
- Modify: `D:/ind-session/ACTIVE_WORK_CONTEXT.md`

- [ ] Run page 77, 79, 88, and 89 overview regressions.
- [ ] Run relevant Markdown/export deterministic tests.
- [ ] Run py_compile and touched-file diff checks.
- [ ] Record the generalized parent-header decision and synchronize mirror files.

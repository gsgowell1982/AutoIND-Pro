# Toxicokinetic Overview Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an eight-column toxicokinetic overview profile and render its source-backed location parent header.

**Architecture:** Keep the profile typed and separate, reconstruct its canonical leaves from word geometry, then reuse shared overview parent-group, row projection, and Markdown machinery.

**Tech Stack:** Python, PDF word geometry, semantic table AST, unittest.

---

### Task 1: Establish page 91 failure

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Assert page 91 has the new profile and eight canonical leaves.
- [ ] Assert `位置` spans `卷/页码` and Markdown renders two header rows.
- [ ] Assert the wrapped first study type is merged.
- [ ] Run the focused test and confirm the profile is currently absent.

### Task 2: Add typed schema and admission

**Files:**
- Modify: `parsers/pdf/postprocess.py`

- [ ] Define the eight canonical columns and profile id.
- [ ] Admit only tables with the toxicokinetic axis signature and locator evidence.
- [ ] Include the profile in overview projection recognition.

### Task 3: Reconstruct headers and rows

**Files:**
- Modify: `parsers/pdf/postprocess.py`

- [ ] Build the first six anchors from the main header band.
- [ ] Build `卷/页码` anchors from the lower band.
- [ ] Reuse the dynamic location parent group.
- [ ] Add eight-column useful-row and wrapped-first-cell behavior.

### Task 4: Focused verification and synchronization

**Files:**
- Test: `tests/parser_tests/test_r2_regression.py`
- Modify: `D:/ind-session/ENGINEERING_DECISIONS.md`
- Modify: `D:/ind-session/ACTIVE_WORK_CONTEXT.md`

- [ ] Run page 77/79/88/89/91 profile regressions and Markdown tests.
- [ ] Run py_compile and touched-file diff checks.
- [ ] Record the typed schema decision and synchronize mirror files.

# Template Source Visual Row Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve physical template row order after late adjacent-row ownership attachment.

**Architecture:** Store complete source-instance visual row records in the inline projection and make Markdown replace consumed source IDs in that physical stream. Keep the current signature renderer as a compatibility fallback.

**Tech Stack:** Python, parser semantic projection, Markdown renderer, unittest regression tests.

---

### Task 1: Establish failing order tests

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Add a unit test where duplicate text instances are distinguished by source ID and a projected row occupies its consumed source position.
- [ ] Extend page 65 regression to require the dose/control row first and before female toxicokinetics in Markdown.
- [ ] Run both tests and confirm the current append-order behavior fails.

### Task 2: Persist canonical source visual rows

**Files:**
- Modify: `parsers/pdf/postprocess.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Build source visual row records from physically sorted owned text nodes during inline projection.
- [ ] Mark whether source-instance coverage is complete.
- [ ] Preserve projected row source IDs and bounding boxes.

### Task 3: Render by source instance

**Files:**
- Modify: `api/main.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Use complete source visual rows as the primary Markdown ordering stream.
- [ ] Insert each projected row at the earliest consumed source ID.
- [ ] Retain the signature-based fallback for older projections.

### Task 4: Focused verification and mirror synchronization

**Files:**
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] Run page 64-69 inline-template regressions and unit projection tests.
- [ ] Run Python compilation and `git diff --check`.
- [ ] Synchronize the five changed files to the mirror and compare SHA-256 hashes.

# Source-Backed Body Span Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve source merged body cells in reviewer presentation while keeping semantic rows fully populated and real source repetitions unchanged.

**Architecture:** Convert validated parser row groups into canonical `presentation_spans`, map those spans through Markdown projection and continuation composition, and activate HTML tables only for source-backed presentation spans.

**Tech Stack:** Python, `unittest`, PDF semantic projection, IND-review Markdown with HTML tables.

---

### Task 1: Lock dual-layer behavior

- [x] Add a positive row-group-to-presentation-span fixture.
- [x] Add a negative repeated-semantic-value fixture without source row-group evidence.
- [x] Add IND-review HTML rowspan assertions.
- [x] Add page-103/page-104 activation and subject rowspan assertions.
- [x] Confirm failures before production changes.

### Task 2: Project source row groups

- [x] Collect established row-group evidence classes.
- [x] Find a unique compatible semantic window for each group.
- [x] Reject ambiguous, out-of-range, or overlapping groups.
- [x] Store canonical `presentation_spans` and coverage metadata.

### Task 3: Compose and render spans

- [x] Map local presentation spans through projected-grid row provenance.
- [x] Rebase spans across continuation row suppression and merged offsets.
- [x] Extend trailing spans only over inherited omitted-dimension prefixes.
- [x] Render IND-review HTML only when validated presentation spans exist.

### Task 4: Verify and synchronize

- [x] Run deterministic Markdown and parser contracts.
- [x] Run focused cross-type r2 row-group, continuation, and header regressions.
- [x] Run syntax and whitespace checks.
- [x] Record the engineering decision and risk-based regression scope.
- [x] Synchronize files and documents to the mirror and run focused mirror verification.

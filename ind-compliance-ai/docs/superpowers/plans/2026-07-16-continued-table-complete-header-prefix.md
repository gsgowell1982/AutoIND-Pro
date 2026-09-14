# Continued Table Complete Header Prefix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every inherited header layer from continued-table body rows while preserving the root table's complete projected header.

**Architecture:** Treat projected root header rows as a bounded equivalence set. Consume matching rows consecutively from each continuation prefix, with each root layer eligible once, and feed the total skipped count into existing merged-row rebasing.

**Tech Stack:** Python, `unittest`, IND-review Markdown chain composition.

---

### Task 1: Lock the failure

- [x] Add a deterministic parent-plus-leaf continuation fixture.
- [x] Assert merged semantic rows contain only the root header prefix.
- [x] Assert the real page-103 parent and leaf header rows each occur once.
- [x] Confirm deterministic and real tests fail on the inherited leaf row.

### Task 2: Consume the complete header prefix

- [x] Add a helper that consumes only matching continuation-leading rows.
- [x] Remove each matched root header candidate from further consideration.
- [x] Return the complete skipped-row count to existing coordinate rebasing.
- [x] Run deterministic and real RED/GREEN tests.

### Task 3: Verify and synchronize

- [x] Run the complete deterministic Markdown module.
- [x] Run focused pages 102-106 genotoxicity continuation regressions.
- [x] Run syntax and whitespace checks.
- [x] Synchronize files to the mirror, compare hashes, and run focused mirror verification.

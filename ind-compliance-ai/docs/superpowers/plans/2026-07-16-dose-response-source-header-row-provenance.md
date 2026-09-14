# Dose-Response Source Header and Row Provenance Implementation Plan

**Goal:** Keep dose-response Markdown source-faithful while preserving structural rows after presentation-only header projection.

**Architecture:** Separate source-visible header evidence from normalized leaf identity, then resolve structural row metadata through projected-to-source row provenance.

**Tech Stack:** Python, `unittest`, PDF semantic projection, IND-review Markdown.

---

### Task 1: Lock the failures

- [x] Add a fixture proving internal M/F leaves do not authorize a standalone sex row.
- [x] Add a fixture proving `Additional examinations` survives before a merged separator row.
- [x] Add a source-backed sex-row fixture that still exercises inserted-row coordinate rebasing.
- [x] Run the fixtures and confirm the previous implementation fails for the expected reasons.

### Task 2: Separate evidence from presentation

- [x] Record `source_has_explicit_sex_header_row` during ordinary result-panel recovery.
- [x] Carry the same evidence through page-word-only continuation recovery.
- [x] Gate standalone sex-row rendering on the explicit source-evidence field.

### Task 3: Preserve semantic row coordinates

- [x] Derive projected-grid source-row identities from normalized semantic row signatures.
- [x] Consume duplicate signatures in source order.
- [x] Resolve `merged_rows` against projected row identities before rendering.

### Task 4: Verify and synchronize

- [x] Run deterministic RED/GREEN tests and the complete Markdown export module.
- [x] Run focused real r2 page 97-100 regressions.
- [x] Run syntax/whitespace checks, synchronize project files, compare hashes, and run focused mirror verification.

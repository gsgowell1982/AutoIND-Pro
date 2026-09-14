# Source-Aware Continuation Matrix Header Rows Implementation Plan

**Goal:** Preserve repeated labels that belong to distinct same-page template instances and render same-line dose matrix headers as one source-faithful Markdown item.

**Architecture:** Partition continuation content by source block identity and rebuild each template from its remaining owned source nodes. Extend the existing inline form visual projection with a guarded `inline_matrix_header_row` derived from same-line owned geometry and dose-header semantics; keep normalized text matching only as a presentation-consumption mechanism.

**Tech Stack:** Python, existing PDF postprocessor, `unittest`, IND-review Markdown renderer.

---

### Task 1: Lock The Regression

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Add a synthetic same-page split test with duplicate parent/child labels backed by distinct source IDs.
- [ ] Assert the child receives only the later source instances and the parent retains the earlier instances.
- [ ] Add a page-69 regression for source ownership, matrix-header projection, Markdown grouping, and physical order.
- [ ] Run the focused tests and observe RED before production edits.

### Task 2: Make Continuation Partition Source-Aware

**Files:**
- Modify: `parsers/pdf/postprocess.py`

- [ ] Remove moved source IDs from the parent ownership set.
- [ ] Rebuild the parent rows and fields from remaining owned source nodes.
- [ ] Preserve source-less synthetic items only through an occurrence-count fallback when they cannot be reconstructed from source nodes.
- [ ] Verify repeated normalized text in separate source instances survives in both parent and child templates.

### Task 3: Project Same-Line Matrix Headers

**Files:**
- Modify: `parsers/pdf/postprocess.py`
- Verify: `api/main.py`

- [ ] Detect owned same-line non-colon cells only when the leading cell has generic dose/matrix-header semantics and the row occupies distinct non-overlapping columns.
- [ ] Reject titles, notes, instructions, prose, result rows, and nodes crossing a continuation boundary.
- [ ] Append an `inline_matrix_header_row` to the existing inline form projection with source IDs, cell geometry, display text, and exact-once consumption metadata.
- [ ] Confirm the existing Markdown projection consumer emits the combined row in source order.

### Task 4: Focused Verification And Mirror Sync

**Files:**
- Verify: `parsers/pdf/postprocess.py`
- Verify: `api/main.py`
- Verify: `tests/parser_tests/test_r2_regression.py`
- Mirror: `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai`

- [ ] Run synthetic continuation and inline projection tests.
- [ ] Run real regressions covering pages 54-56, 64-71, same-page tail-note splitting, page-bottom relinking, and Markdown note safety.
- [ ] Audit all generated matrix-header projections in r2 for geometry and ownership invariants.
- [ ] Run `py_compile` and `git diff --check`.
- [ ] Apply the scoped patch to the mirror, rerun focused verification, and confirm hashes match.

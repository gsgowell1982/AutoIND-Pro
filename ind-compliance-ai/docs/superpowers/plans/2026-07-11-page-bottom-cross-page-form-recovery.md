# Page Bottom Cross-Page Form Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover IND form instances whose title and first metadata row occur at the bottom of one page while the remaining form body begins on the next page.

**Architecture:** Build a conservative page-bottom template-head fragment from a strong numbered IND form heading, nearby local title, and a same-row field lattice with at least two field anchors. After all current-page template detectors run, link the topmost compatible untitled template to the preceding fragment without duplicating ownership; use the existing mixed-row projection for the fragment's heterogeneous first row.

**Tech Stack:** Python, PDF postprocessor, structure-template AST, `unittest`, IND-review Markdown renderer.

---

### Task 1: Lock Real Document Behavior

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] Assert page 67 renders `design question + dosing time + study number` as one mixed row.
- [ ] Assert page 68 contains both the prior matrix continuation and a new bottom template-head fragment owning `txt_p68_018` through `txt_p68_022`.
- [ ] Assert the page 68 fragment projects `txt_p68_020/021/022` into one mixed row while preserving the source AST role.
- [ ] Assert the page 69 top template is linked with `continued_from_structure_template_id`, `continued_from_page`, and the prior title.
- [ ] Assert Markdown contains one heading, one local title, and one complete first-row list item, with no fused body paragraph.
- [ ] Run the selected tests and observe RED at the missing fragment/continuation/companion assertions.

### Task 2: Recover Strong Page-Bottom Template Heads

**Files:**
- Modify: `parsers/pdf/postprocess.py`

- [ ] Add `_build_page_bottom_tabular_form_header_fragments` next to the text-form builders.
- [ ] Seed only from a numbered non-continuation IND heading in the lower page region with report-title, test-article, or study-domain evidence.
- [ ] Collect only nearby, unowned, non-page-chrome text through the page bottom and stop at another heading or a large vertical gap.
- [ ] Require at least two field-like nodes in one visual row, at least one non-field companion in that row, a distinct-column non-overlap lattice, and a local non-field title/section before the row.
- [ ] Build a normal tabular-form template with `detection_source=page_bottom_template_header_fragment`, preserve source IDs/bboxes, and record an explicit fragment signal.

### Task 3: Relink The Next Page And Generalize Mixed Rows

**Files:**
- Modify: `parsers/pdf/postprocess.py`

- [ ] After current-page template detectors finish, find the topmost untitled tabular form when the immediately preceding global template is a page-bottom fragment.
- [ ] Link the existing template in place; set continuation metadata and signals without cloning or re-owning nodes.
- [ ] Keep the current generic template detector as the body owner so its richer rows and notes remain unchanged.
- [ ] Remove `entries/sections` as a mandatory mixed-row companion gate; retain ownership, `row_texts`, title/note exclusions, colon/matrix rejection, y overlap, learned column, unused column, and horizontal non-overlap guards.

### Task 4: Verify And Mirror

**Files:**
- Verify: `parsers/pdf/postprocess.py`
- Verify: `api/main.py`
- Verify: `tests/parser_tests/test_r2_regression.py`
- Mirror: `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai`

- [ ] Audit all r2 page-bottom fragments, relinked continuations, and row companions; every new object must satisfy the generic evidence contract.
- [ ] Run page 63-72 template, mixed-row, note-order, matrix, and Markdown regressions.
- [ ] Run Python compilation and `git diff --check`.
- [ ] Apply the scoped parser/test changes to the mirror, rerun focused tests, and verify SHA-256 equality.

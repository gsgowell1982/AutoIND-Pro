# Zero Regression Baseline Design

## Goal

Reduce the current `test_r2_regression` failure set from five to zero by fixing the one real note-order defect and replacing four representation-bound assertions with stable semantic contracts.

## Evidence

The five failures split into two classes:

- Page 24 contains a real source-order defect. A first note fragment is extracted from the final raw table row while its continuation is already owned as a positioned below-table text block. The extracted fragment is appended without page or position metadata, so Markdown sorts the positioned continuation first.
- The two architecture assertions, the page 37 list-prefix assertion, and the page 101 synthesized sex-row assertion no longer match current contracts. Their semantic content is present and correctly owned.

## Approaches Considered

### 1. Source-order lineage plus semantic assertions (selected)

Attach raw-row identity and source-order coordinates when a trailing row becomes a note. Let Markdown compare actual bbox coordinates with explicit source-order coordinates. Change tests to validate ownership, source reasons, visible source evidence, and label-line policy rather than sequential IDs or invented rows.

This preserves evidence and generalizes to any note split across a table boundary.

### 2. Renderer-only source ranking

Always render `trailing_table_note_row` before `below` notes. This is smaller, but it encodes a coarse relation rule without retaining where the extracted row came from and is harder to audit across pages.

### 3. Text repair

Teach Markdown to rewrite reversed fragments such as `测 ... -未检`. This is rejected because it is language- and value-specific and cannot recover arbitrary split notes.

## Data Contract

An extracted trailing note adds:

- `source_grid: raw_grid`
- `source_row_number`: one-based original raw row
- `source_row_ref`: `<table_id>:raw_row:<number>`
- `physical_page`: owning table page
- `source_order_y`: owning table bbox bottom
- `source_order_x`: owning table bbox left

The order coordinates are not claimed as the note's exact bbox. They express the proven ordering boundary: the row was inside the table and preceded separately captured `below` content.

Markdown uses a positioned segment's bbox when available. For bbox-less extracted rows, it uses `physical_page/source_order_y/source_order_x`. Original index remains the final tie-breaker.

## Stable Regression Contracts

- Architecture source IDs must resolve to absorbed source objects whose target owner is in the study render plan; ordinal template numbers are not asserted.
- A template note-group label renders as a literal line before its following remark, without a Markdown list prefix.
- A dose-response sex display row is emitted only when `source_has_explicit_sex_header_row` is true. Internal `dose + sex` leaf identity remains unchanged.
- Page 24 renders the complete note in physical order and contains `-未检测` exactly as source fragments compose.

## Scope

Modify only:

- `parsers/pdf/table_modules/postprocess.py`
- `api/main.py`
- `tests/parser_tests/test_r2_regression.py`
- project design/plan and engineering-decision documentation

No page numbers, table IDs, exact note strings, template ordinals, or fixed coordinates may enter production logic.

## Verification

- New unit test proves extracted trailing notes retain lineage and sort before positioned below-table continuations.
- The five formerly failing tests pass together.
- Page 24, page 37, page 101, architecture, page 106, page 109, and page 114 focused regressions pass.
- Complete `tests.parser_tests.test_r2_regression` passes with zero failures.


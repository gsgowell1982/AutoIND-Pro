# Source-Aware Result-Matrix Trailing Colspan Design

## Goal

Preserve trailing result-matrix metadata rows such as study numbers and CTD locations, including their centered multi-group spans, when logical cells do not carry usable geometry.

## Evidence Model

Trailing rows use a shared physical-row adapter:

- `display_grid` identifies expected trailing row labels and protects source completeness;
- table text atoms provide preferred word geometry;
- page words inside the current table bbox fill missing atom rows;
- legacy cells remain a fallback only when they carry valid bboxes.

Page words are deduplicated against table atoms by normalized text and rounded bbox. Physical rows are clustered by y. Label fragments are separated from value fragments using the first leaf-column boundary, and adjacent value fragments are merged before span assignment.

## Span Projection

The existing repeated-leaf schema supplies leaf centers, group count, and leaves per group. When two centered values cover four condition groups, each value spans two groups. For a 12-leaf matrix this produces spans `1..6` and `7..12`.

Each projected row retains its normalized label, source bbox, source kind, and span records. CTD word fragments are merged into complete location values before projection.

## Commit Protection

Expected trailing labels are collected from the source display grid. A study-context semantic binding is committable only when every expected trailing label has a projected row with spans. If coverage is incomplete, the binding is rejected and diagnostics are emitted, preserving the existing table surface instead of silently replacing it with an incomplete semantic grid.

## Regression Protection

- r2 page 86 projects `95102` over leaves 1-6 and `95156` over leaves 7-12.
- The two CTD locations receive the same spans.
- Both rows appear after the last data row and before `附加信息：` in Markdown.
- Page-85/86 note ownership, preludes, grouped condition headers, and page-81 parent headers remain unchanged.

## Scope

Changes are limited to study-context result-matrix trailing-row evidence, binding commit validation, orchestration of page-word evidence, and focused tests. Production code contains no page, document, value, or coordinate-specific rules.

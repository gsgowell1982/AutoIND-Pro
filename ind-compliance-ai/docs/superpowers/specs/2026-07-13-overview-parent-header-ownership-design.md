# Overview Parent Header Ownership Design

## Problem

The page-77 overview table correctly projects `位置` as a parent header spanning
the `卷` and `页码` leaf columns. The same source text remains an unowned AST body
block because it sits just above the detected table bounding box, so Markdown
renders it again as standalone prose.

## Decision

Preserve source geometry for multilevel parent headers produced by the overview
inventory word projection. The existing table-cell ownership closure will use
that explicit semantic-header region, in addition to the physical table box, to
claim the matching AST text block.

Binding requires both:

1. Equivalent normalized text between the AST block and semantic parent header.
2. Strong geometric overlap with the preserved parent-header bounding box.

When bound, the source block is added to `owned_text_block_ids`, recorded in
`header_source_block_ids`, and marked `business_table_header` with the owning
table ID. The semantic header group and span-cell records retain the source ID
for auditability.

The later composite-object reconciliation may remove that owned source block
from the page's visible AST flow. Source provenance remains available on the
table, header group, and span cell, while the standalone block is no longer a
render candidate.

## Rejected Alternatives

- Markdown text deduplication is unsafe because `位置` is a common legitimate
  label and loses source ownership semantics.
- Expanding every table bounding box upward can absorb section headings, study
  metadata, or notes that do not belong to the table.
- A page-77 or literal-`位置` condition would not generalize to other overview
  tables or other parent-header labels.

## Regression Protection

A focused ownership test will include one matching parent-header block and one
same-text block at a different location. Only the geometrically supported block
may be claimed. The page-77 full-PDF regression will require the source block to
be owned by `tbl_021`, preserve the two-level `位置 -> 卷/页码` table header, and
exclude standalone `位置` from the section Markdown.

Verification is limited to the focused ownership test, page 77, nearby overview
inventory tables, syntax compilation, diff hygiene, and mirror parity. This
change does not affect general PDF extraction or unrelated table families.

# Cross-Owner Tail and Prelude Boundary Design

## Goal

Preserve a new study object's `示例` prelude when the same visual grid begins with notes owned by the preceding table.

## Boundary Model

A mixed visual study grid has three independent boundaries:

1. previous-owner tail notes;
2. next-object prelude rows;
3. the next study object's formal section title.

`示例` remains a hard boundary for result-matrix recovery, but it is not the formal study-section start when the next meaningful row is a valid study title. The mixed-owner transfer path must use the formal title as `section_start`, allowing leading-row classification to assign preceding note rows to the previous table and the prelude to the next structure template.

## Evidence Model

Prelude geometry is resolved from table word evidence first. If the table evidence omits a standalone prelude, page-level words are clustered into physical rows and used as a fallback. The resulting prelude retains bbox, source row reference, source table, and destination template ownership before source-grid cropping.

## Commit Rule

Cropping remains conditional on an evidence-complete destination template. The transfer plan must explicitly represent every meaningful row between the prior note tail and the new title; a prelude must not be silently consumed as a generic boundary.

## Regression Protection

- r2 page 86 template and page AST own one `示例` prelude with bbox.
- `tbl_031` records `transfer_to_next_object_prelude` for that row.
- Markdown order is page-85 note group, `示例`, page-86 title, page-86 metadata/table.
- Existing page-85 prelude and page-85-to-86 note ownership remain unchanged.

## Scope

The change is limited to visual study-panel boundary selection, page-word evidence adaptation, and focused r2 tests. No page, compound, exact coordinate, or exact note wording is used in production rules.

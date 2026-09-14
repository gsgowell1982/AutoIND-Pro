# Source-Backed Body Span Presentation Design

## Problem

Typed semantic grids carry group keys forward so every data row is independently meaningful. IND-review Markdown currently renders those values repeatedly, even when the source table contains one merged body cell. Other tables physically repeat the value on every row. Treating both forms the same loses source fidelity; inferring merges from repeated semantic values alone would incorrectly collapse real repetitions.

## Decision

Maintain separate contracts:

- `semantic_grid` remains row-complete and machine-oriented.
- `presentation_spans` contains only validated source-backed logical row/column spans.
- IND-review uses HTML tables when `presentation_spans` is present, preserving `rowspan` and `colspan` while leaving the semantic grid unchanged.
- Tables without validated presentation spans remain pipe tables with complete semantic values.

## Source Row-Group Conversion

Collect established `row_groups` from the table and grouped borderless projection. A group is promoted only when:

- its source is an established body-rowspan evidence class;
- it has a positive logical column and `rowspan >= 2`;
- exactly one contiguous window in the final semantic grid is compatible with the group text;
- every populated covered cell equals or safely contains the group text;
- it does not overlap a stronger accepted span.

Coordinates in `presentation_spans` are zero-based against the canonical semantic grid.

## Continuation Composition

Markdown chain composition maps each table's semantic span coordinates through the projected-grid row provenance already used for structural rows. If a trailing root span and an inherited-schema continuation have the same semantic value in the same column, the continuation source omits that logical column, and the equal values form a consecutive continuation prefix, extend the span only across that prefix.

## Safety

- Never create spans from repeated semantic values alone.
- Never blank semantic cells to imitate source merging.
- Preserve tables that physically repeat labels as repeated presentation.
- Require unique source-group-to-semantic-window mapping.
- Bound continuation extension to inherited schemas with source-width evidence and stop at the first value change.
- Keep presentation spans separate from logical data and header spans.

## Verification

- Deterministic positive source-row-group conversion.
- Deterministic negative repeated-value table without source group.
- Deterministic IND-review HTML rowspan rendering.
- Deterministic continuation-prefix span extension.
- Real r2 page-103/page-104 activation and subject rowspans.
- Focused cross-type r2 tables with existing row-group and header behavior.

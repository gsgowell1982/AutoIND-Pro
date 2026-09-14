# Overview Dynamic Location Parent Header Design

## Goal

Project source-backed `位置` or `CTD中的位置` parent headers over adjacent `卷` and `页码/部分` leaves for every overview inventory profile, while preserving existing PK tables and continuation schemas.

## Semantic Model

Overview inventory schemas keep two separate surfaces:

- `leaf_columns`: the canonical data-column names used for row projection and continuation inheritance;
- `header_column_groups`: typed parent spans used to build the two-row presentation header.

The rendered semantic grid may repeat the parent text across its leaf columns, but it is not the canonical leaf schema.

## Detection

The detector searches canonical leaf columns for an adjacent pair `卷` plus `页码` or `部分`. It then requires a source word above or within the header band whose normalized text is `位置`, `CTD中的位置`, or `CTD位置`. The parent center must fall within the child-anchor span tolerance. Column indexes are derived from the leaf pair and are not profile constants.

No parent is synthesized from leaf names alone. This protects flat tables that have `卷/页码` leaves but no source-backed parent label.

## Continuation

A continuation table inherits the parent's projection `leaf_columns` and `header_column_groups`. It must not use `parent.semantic_grid[0]` as the leaf schema because that row contains repeated parent labels after multilevel projection. Source-backed groups detected locally override inherited groups only when they validate against the same leaf schema.

## Presentation

The page 88 table presents:

```text
... | 试验编号 | 位置 | 位置
... |          | 卷   | 页码
```

The span metadata records `start_leaf_col=8`, `end_leaf_col=9`, and `colspan=2`. Page 89 inherits the same two header rows and span metadata.

## Regression Scope

- Page 77 and page 79 PK overview parent headers remain unchanged.
- Page 88 nonclinical overview gains the location parent span.
- Page 89 continuation inherits canonical leaves and the parent group.
- A synthetic flat overview with `卷/页码` but no parent word remains single-level.
- Existing data-row compaction and marker-note ownership remain unchanged.

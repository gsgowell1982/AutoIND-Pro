# Study Metric Grouped Header Markdown Design

## Problem

`study_metric_grouped_matrix_projection` keeps canonical leaf columns and body rows in `semantic_grid`, while source-backed parent spans live in `header_column_groups`. IND-review Markdown selected the semantic grid but had no adapter that materialized those parent rows. As a result, r2 page 92 retained its AUC values but lost the `稳态AUC`, `小鼠a`, and `大鼠b` hierarchy.

## Decision

Keep `semantic_grid` as the canonical leaf-and-data contract. Add a presentation adapter for the typed `study_metric_grouped_matrix` profile and delegate row construction to a generic grouped-header materializer. The materializer consumes `row`, `start_col`, `end_col`, and `text`, validates bounds and non-overlap, then emits ordered parent rows plus the leaf row.

Markdown approximates a colspan by repeating the parent label across its covered columns, matching the existing presentation convention. Ungrouped terminal headers are placed at the deepest parent level; leaf labels below multi-column groups remain in a separate leaf row.

## Safety Gates

- Require the exact typed semantic profile.
- Require at least one valid multi-column span.
- Reject malformed, out-of-bounds, or overlapping groups without changing the input grid.
- Detect an already-materialized group surface from the authoritative group coordinates.
- Do not use page, title, compound, metric wording, or fixed column-count rules.

## Verification

- Deterministic three-level AUC fixture.
- Deterministic no-span negative fixture.
- Real r2 page 92 final Markdown assertions.
- Real r2 page 94 single-parent/multiple-leaf generalization assertions.
- Focused protection checks for page 85 and page 91 header projection families.


# Table Semantic Projection v2 Design

## Goal

Improve generalized table parsing for IND application materials by separating table region discovery from logical table structure projection. OpenDataLoader 200-sample results are used as diagnostic evidence, but the production target remains enterprise IND/eCTD parsing with auditable evidence, no PDF/page/sample/text hardcoding, and no runtime dependency on external parser projects.

## Problem

The current framework usually finds table-like regions, but several weak samples show that the detected region is not yet projected into the correct logical table:

- visual item lists inside one cell are exploded into multiple rows;
- flowchart connector or arrow columns are merged into neighboring text;
- projected row-header or stub columns are lost;
- rowspans and grouped body rows are flattened or over-compressed;
- repeated boundary rows can cause adjacent fragments to stay split;
- Markdown pipe tables cannot preserve rowspan/colspan semantics for benchmark-style topology scoring.

This is a structure-projection issue, not primarily a detector issue.

## Architecture

Add a focused table semantic projection layer after the existing table AST is built and before final Markdown/export consumption. The layer consumes current parser evidence: `raw_grid`, `display_grid`, `data_grid`, `header`, `cells`, `bbox`, source/detection metadata, note blocks, header groups, row groups, and continuation metadata. It emits additive semantic metadata and, only when evidence is strong, a corrected public grid:

- `table_family`
- `semantic_projection_v2`
- `semantic_grid`
- `semantic_header`
- `logical_cells`
- `header_column_groups`
- `header_row_groups`
- `row_groups`
- `projection_diagnostics`

The layer does not replace raw evidence. `raw_grid` remains the audit source.

## Stage 1.1 Scope

Stage 1.1 targets `table_teds_s/table_quality_mean >= 0.85` without weakening protected IND regressions. It covers four generalized families:

1. `two_column_inventory`: list-in-cell tables where section labels, item lists, and repeated boundary rows should remain one logical two-column table.
2. `flowchart_matrix`: compact schema rows with connectors/arrows and downstream structured rows; body prose must not become rows.
3. `comparison_matrix`: left projected row-header/stub column plus two or more comparison columns.
4. `rowspan_grouped_table`: sparse leading categorical cells that visually span following continuation rows.

The first implementation may add semantic metadata for all four families, but it should rewrite `display_grid/data_grid` only for high-confidence cases where doing so improves structure without losing text evidence.

## Export Strategy

Customer-facing Markdown stays readable and conservative. Benchmark/diagnostic export can later render semantic tables as HTML when span metadata exists, because Markdown pipe tables cannot express rowspans and colspans. Stage 1.1 records the semantic metadata first; HTML export can be added in Stage 1.2 if metrics show Markdown topology is the remaining bottleneck.

## Test Strategy

Tests are written before implementation:

- regression tests for OpenDataLoader weak samples `01030000000121`, `01030000000120`, `01030000000182`, and one structure-projection sample such as `01030000000200`;
- deterministic tests for metric report fields `table_teds_mean`, `table_teds_s_mean`, and `table_quality_mean`;
- protected gates for A-tst, r2, 2-column-tst, eCTD, table header grammar, and OpenDataLoader regression tests.

## Non-Goals

- No PDF filename/page/sample-id/text-specific production branch.
- No direct integration of OpenDataLoader, MinerU, or their outputs into production parser runtime.
- No promise that all metrics reach `1.0`; OCR, formula recognition, chart semantics, and annotation-style differences remain separate framework layers.

# Study Condition Matrix Source Header Design

## Goal

Preserve the source-backed row order and header semantics of IND study-condition result matrices such as r2.pdf page 86, without introducing renderer-authored labels such as `条件/时间`.

## Problem

The parser correctly recovers the repeated leaf columns and the `时间` row axis, but the semantic projection collapses them into one header row and drops the source stub `排泄途径(4)`. The Markdown renderer then prepends a synthetic `条件/时间` row before the study descriptor rows.

## Design

Keep the existing 13-column result `semantic_grid` for compatibility with downstream consumers. Add explicit `matrix_header_rows` to `study_condition_grouped_result_matrix_projection`:

- `matrix_leaf_header`: the source-backed stub such as `排泄途径(4)` plus the repeated leaf columns.
- `matrix_row_axis`: the source-backed row-axis stub such as `时间` plus blank value cells.

Recover the leaf-header stub from the same visual row as the repeated leaf atoms, using geometry rather than page numbers or fixed Chinese labels. Preserve the text exactly, including note markers such as `(4)`.

For a bound composite table, Markdown output order is:

1. Study descriptor rows beginning with `种属`.
2. Source-backed matrix leaf-header row.
3. Source-backed matrix row-axis row.
4. Result body rows.
5. Trailing colspan rows and note rows already present in the semantic grid.

The renderer must not generate `条件/时间`. If explicit matrix header rows are unavailable, retain the existing flattened fallback for older or incomplete projections.

## Regression Protection

- Unit-test the Markdown projection order and absence of `条件/时间`.
- Update page 86 integration assertions to require `种属` first, retain `排泄途径(4)`, and render `时间` as a separate blank-axis row.
- Verify the structurally similar page 87 projection.
- Run focused study-condition matrix tests and the nearby pages 85-87 regression slice rather than the entire PDF suite unless focused verification reveals broader impact.

# Canonical Composite Table Surface Design

## 1. Goal

Represent composite IND tables as source-faithful structured tables without weakening the normalized logical data used by downstream rules, search, and comparison.

The immediate evidence is the grouped study-condition tables on r2 pages 86 and 87. The design must generalize to real IND tables with variable condition-group counts, repeated subcolumns, multi-line descriptor values, and mixed row/column spans. It must not introduce page-, section-, title-, or table-ID-specific behavior.

## 2. Problem Statement

The current parser correctly extracts the central result matrix, but separates the preceding study-condition template from that matrix. The API later reconstructs a composite Markdown table by repeating each condition descriptor across the group's leaf columns.

This creates three architectural defects:

1. The AST does not contain the complete table surface that is rendered.
2. Real source spans in descriptor and metadata rows are replaced by repeated text.
3. Multi-line descriptor values are assigned from flattened text order after source x-coordinates have been lost.

On page 86, that third defect assigns `胶囊` to the fourth condition group instead of the third. The source geometry places `胶囊` in group 3 and `溶液 / 生理盐水` in group 4.

Pages 111, 112, and 113 provide negative controls: legacy structural hints suggest spans that are not present in the source PDF. Those hints must remain non-authoritative and must not create canonical spans.

## 3. Design Principles

- Preserve normalized logical data and source-faithful display structure as separate table surfaces.
- Keep one canonical span authority: `cell_spans`.
- Bind source content by geometry before relying on flattened reading order.
- Create canonical structure only from source-backed evidence.
- Fail closed when geometry is ambiguous: preserve extracted content, emit diagnostics, and avoid inventing spans.
- Keep all recognition rules structural and content-agnostic.

## 4. Table Data Model

### 4.1 Logical Surface

`semantic_grid` remains the normalized logical result matrix. Existing rule, search, comparison, and validation consumers continue to use it unchanged.

For a grouped study-condition table, this surface contains the matrix headers and result rows, but does not repeat the preceding descriptor template.

### 4.2 Display Surface

`semantic_display_grid` is added when the source table is a composite of a condition template, a result matrix, and optional trailing metadata rows.

It contains, in source order:

1. condition descriptor rows;
2. matrix header rows;
3. matrix body rows;
4. trailing report or CTD-location rows.

Each cell retains provenance sufficient to trace its text and structural decision to source blocks, words, bounding boxes, or parser evidence.

`semantic_display_data_start_row` records the first body row of the result matrix within `semantic_display_grid`. It is derived from constructed row roles, not from page-specific offsets.

### 4.3 Canonical Spans

`cell_spans` remains the only authoritative representation of rowspan and colspan. Each span declares its coordinate space:

```json
{
  "coordinate_space": "semantic_display_grid",
  "row": 0,
  "col": 1,
  "rowspan": 1,
  "colspan": 3,
  "role": "header",
  "source": "study_condition_geometry"
}
```

Supported coordinate spaces are:

- `semantic_grid`
- `semantic_display_grid`

A rendered table surface must use spans from exactly one coordinate space. Legacy metadata may remain as evidence or diagnostics, but it cannot independently control rendering.

## 5. Composite Table Recognition

A table qualifies for composite projection only when structural evidence establishes all of the following:

1. a descriptor template immediately associated with a result matrix;
2. two or more condition groups;
3. a repeated leaf-header pattern with a stable period of at least two columns;
4. compatible horizontal extents between the template groups and matrix groups;
5. source-order continuity without an intervening unrelated table or section boundary.

The recognizer must infer group count and leaf width dynamically. It must not depend on known labels such as `尿液`, `粪便`, `合计`, known section numbers, or fixed column counts.

Confidence is reduced by conflicting group counts, unresolved group boundaries, overlapping unrelated blocks, or incomplete source geometry. Below the canonical threshold, the parser keeps the logical matrix and records a review diagnostic instead of constructing a display surface.

## 6. Geometry-Driven Group Binding

### 6.1 Group Intervals

Condition-group x-intervals are derived from the repeated matrix leaf columns. The preferred evidence order is:

1. source cell boundaries;
2. word bounding boxes grouped by leaf-column centers;
3. stable inferred column centers from matrix rows.

The algorithm supports unequal group widths when the source geometry proves them. Equal-width partitioning is only a fallback when the matrix itself provides a regular repeated pattern.

### 6.2 Descriptor Row Bands

Descriptor rows are identified from positioned words within the associated template bounding box. Row bands are formed by vertical overlap and baseline proximity, preserving source y-order.

The label region is separated from condition-value regions using the first group boundary. Labels are not identified from a fixed vocabulary; text role and left-side alignment are the primary evidence.

### 6.3 Token Assignment

Each value token is assigned to the group whose x-interval contains the token center or has the strongest horizontal overlap. Nearest-center assignment is allowed only within a bounded tolerance.

Multi-line values assigned to the same descriptor row and group are joined in source y-order. This preserves cases such as:

- group 1: `溶液 / 水`
- group 2: `溶液 / 生理盐水`
- group 3: `胶囊`
- group 4: `溶液 / 生理盐水`

Flattened `row_texts` may be used only when the number and order of values are unambiguous for the inferred group count. If a continuation line cannot be assigned uniquely, the parser must not guess.

## 7. Display Surface Construction

### 7.1 Descriptor Rows

Each descriptor row contains one label cell followed by one value cell per condition group. A condition value spans the group's leaf columns:

```text
label | group 1 value (N leaves) | group 2 value (N leaves) | ...
```

Adjacent groups with identical text remain distinct cells and distinct spans. Text equality is never sufficient evidence for merging groups.

### 7.2 Matrix Rows

Existing logical matrix headers and body rows are inserted without altering their logical ordering. Canonical matrix spans are translated into display-grid row coordinates when necessary.

Repeated leaf headers remain separate leaf cells. A source group header spanning those leaves is represented by a canonical colspan in the display coordinate space.

### 7.3 Trailing Metadata Rows

Report-number, CTD-location, and similar trailing rows use their measured source horizontal ranges. A metadata value may span one or more condition groups when the source cell boundary proves that relationship.

Centered text, blank neighboring cells, or repeated content alone cannot create a colspan.

### 7.4 Provenance

Every constructed display cell records:

- contributing source block or word identifiers;
- source bounding box when available;
- binding method;
- confidence and diagnostics when fallback evidence was used.

## 8. Rendering Contract

The renderer selects `semantic_display_grid` when it exists and passes only `semantic_display_grid` spans to the canonical table renderer.

Tables containing canonical rowspan or colspan render as semantic HTML. Tables without spans may continue to use pipe Markdown when valid.

The API must not reconstruct grouped study-condition rows by repeating values once a canonical display surface exists. The existing temporary composite projection remains only as a compatibility fallback for old AST payloads that lack `semantic_display_grid`.

The AST and rendered output therefore share the same structural source of truth.

## 9. Failure and Compatibility Behavior

- Existing `semantic_grid` consumers remain compatible.
- Existing canonical spans on ordinary tables remain unchanged.
- Legacy hints such as `header_column_groups`, `row_group`, or inferred blank-cell merges are evidence only.
- A legacy hint becomes canonical only after validation against source geometry and the selected grid.
- Invalid, overlapping, out-of-bounds, or contradictory spans are rejected before rendering.
- Ambiguous composite recognition leaves the logical table intact and emits `review_required` diagnostics.
- Pages 111, 112, and 113 must continue to reject their source-inconsistent legacy span hints.

## 10. Strong-Generalization Invariants

The implementation is acceptable only if all of these remain true:

1. No condition uses a PDF page number, CTD section number, table ID, report number, title string, or expected cell text.
2. Group count and group width are inferred from source structure.
3. Descriptor labels and values may be multi-line and may differ across documents.
4. Identical adjacent values do not collapse distinct source cells.
5. Blank cells do not create spans without corroborating boundaries or geometry.
6. Source x-position takes precedence over flattened token order for group assignment.
7. Logical and display surfaces can evolve independently without duplicating span authorities.
8. Every canonical span is valid, non-overlapping, in bounds, and traceable to source evidence.
9. Failure to prove a composite structure cannot corrupt the logical result matrix.
10. The same recognizer handles pages 86 and 87 and future tables of the same structural family.

## 11. Validation Strategy

### 11.1 Unit Tests

- Infer variable repeated leaf periods and condition-group counts.
- Bind single-line and multi-line descriptor values by x-position.
- Preserve separate groups when adjacent values are identical.
- Reject ambiguous continuation-line assignment.
- Construct descriptor colspans for variable group widths.
- Translate matrix spans into display-grid coordinates.
- Reject overlapping and out-of-bounds display spans.
- Select spans only from the renderer's active coordinate space.

### 11.2 Regression Tests

- Page 86 contains the complete canonical display grid.
- Page 86 assigns `胶囊` to condition group 3 and `溶液 / 生理盐水` to group 4.
- Page 86 descriptor rows, group headers, and trailing metadata reproduce source-backed colspans.
- Page 87 uses the same generic path and exposes canonical composite spans.
- Pages 111, 112, and 113 do not gain false canonical spans.
- Existing page 102 and later cross-row/cross-column table regressions remain stable.

### 11.3 Suite and Audit Gates

- Run the complete protected r2 regression suite.
- Run deterministic Markdown/export tests.
- Audit all r2 business tables for invalid, overlapping, and unrenderable spans.
- Confirm that every rendered span is represented in the AST and no renderer-only structural expansion remains for newly parsed composite tables.
- Sync the authoritative implementation to the mirror and compare hashes before running focused mirror regressions.

## 12. Migration Plan

1. Add display-surface and coordinate-space validation helpers.
2. Add geometry-backed descriptor extraction and group assignment.
3. Build canonical composite display grids and spans in parser postprocessing.
4. Update the renderer to consume the display surface directly.
5. Retain compatibility fallback for previously persisted AST payloads.
6. Replace tests that assert repeated pipe cells with source-faithful HTML and AST assertions.
7. Run full regressions, the all-table span audit, and mirror verification.

## 13. Non-Goals

- Rewriting the core PDF extraction engine.
- Mutating source PDF content.
- Introducing page-specific repair dictionaries.
- Treating visual whitespace or duplicate text as sufficient span evidence.
- Removing the normalized logical matrix used by downstream compliance logic.
- Maintaining a second renderer-only or metadata-only span system.


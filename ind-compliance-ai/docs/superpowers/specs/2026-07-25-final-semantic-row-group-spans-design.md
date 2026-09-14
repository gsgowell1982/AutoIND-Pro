# Final Semantic Row Group Spans Design

## Goal

Generate source-faithful body rowspans from the final semantic table surface so grouped IND records remain structurally consistent after raw PDF rows are repaired, expanded, compacted, or reordered.

The design must handle variable group keys, variable detail counts, multiline source rows, and borderless tables without page numbers, section numbers, table IDs, batch-number patterns, or expected text values.

## Observed Failure

The r2 page 94 impurity table is first extracted as an unstable physical grid and later rebuilt as a normalized logical table. Body row groups are currently projected before that rebuild.

Consequences:

- `LN125` is initially classified as a header and is absent from the early data grid.
- several detail records are packed into one physical row;
- other detail records are distributed across multiple physical rows;
- early groups record physical row counts;
- the later word-level projection creates one semantic row per detail record;
- the stale physical counts are adapted into `semantic_grid` coordinates without recomputation.

The result is structurally invalid: `94NA103` spans three semantic rows although it owns five detail records, and `96NB101` spans two semantic rows although it owns seven.

## Alternatives

### 1. Rerun the Existing Sparse-Blank Detector

Run `project_rowspan_body_groups()` again after semantic projection.

This is small but insufficient. The existing detector was designed for raw rectangular grids, has no final row provenance contract, and treats blank continuation cells as its primary evidence. Reusing it would increase false-span risk in intentionally sparse IND tables.

### 2. Make Every Semantic Projector Emit Its Own Spans

Add body-span logic separately to impurity, genotoxicity, toxicokinetic, and other table-family projectors.

This can preserve family-specific knowledge but duplicates grouping rules, creates inconsistent behavior, and makes future table families opt in manually.

### 3. Final Semantic Row Group Resolver

Add one generic resolver after all semantic table projections. It consumes the final semantic grid, semantic row provenance, source word geometry, and structural column roles. It emits canonical row groups and spans in the final grid coordinate space.

This is the selected approach because it separates extraction repair from presentation structure and provides one generalized span contract for all table families.

## Processing Order

The table pipeline becomes:

1. extract physical table evidence;
2. run table-family semantic projections;
3. finalize `semantic_grid` and row provenance;
4. resolve semantic row groups;
5. project canonical `cell_spans`;
6. render the selected semantic surface.

Early `row_groups` may be retained as source evidence, but they cannot directly become canonical spans after the semantic grid changes.

## Data Contract

### Semantic Row Provenance

Every final semantic body row used by the resolver must expose lineage:

```json
{
  "semantic_row": 5,
  "source_page": 94,
  "source_word_refs": ["word:..."],
  "source_y_min": 263.4,
  "source_y_max": 271.8,
  "projection_source": "study_metric_grouped_matrix_projection"
}
```

Multiple semantic rows may share a physical source row when the source row contained multiple vertically ordered records. Their y-ranges or ordered word references must remain distinct.

### Semantic Row Groups

The final resolver emits `semantic_row_groups`:

```json
{
  "group_key_col": 0,
  "start_semantic_row": 5,
  "end_semantic_row": 9,
  "rowspan": 5,
  "anchor_text": "94NA103",
  "static_cols": [0, 1, 2, 3, 4],
  "detail_cols": [5, 6],
  "source": "final_semantic_row_group_resolution",
  "confidence": 0.97
}
```

`anchor_text` is data, not a recognition pattern. The resolver does not require batch-like syntax.

### Canonical Spans

Each proven static cell produces a `cell_spans` body entry with:

- `coordinate_space: semantic_grid`;
- a final semantic row and rowspan;
- source word or cell references;
- a stable span-group identifier shared by cells in the same row group;
- evidence and confidence.

The canonical span layer remains the only rendering authority.

## Generic Group Resolution

### Candidate Group-Key Columns

Candidate keys are inferred from structural behavior, not fixed column numbers:

- the column contains nonempty anchors separated by blank continuation runs;
- anchors have source-backed vertical regions;
- later columns contain repeated row-level records during those blank runs;
- the column behaves as a low-frequency group dimension relative to detail columns;
- header semantics, when available, support a group or identifier role.

The first column is preferred when evidence is otherwise equal, but it is not hardcoded as the only possible group-key column.

### Group Boundary

A group begins at a nonempty key cell and ends immediately before the next nonempty key cell or the end of the body.

The candidate is accepted only when:

1. it contains at least two final semantic rows;
2. every continuation row is blank in the key column;
3. every continuation row contains content in at least one proven detail column;
4. detail cells have compatible source y-order inside the group's source region;
5. no section, note, subtotal, header, or unrelated row boundary occurs inside the run;
6. the group does not conflict with another accepted group in the same coordinate space.

### Static and Detail Columns

Within an accepted group:

- a static column has a value on the anchor row and blank continuation cells;
- a detail column contains row-level values across the group;
- a column with contradictory nonempty continuation values is not spanned;
- placeholder values and intentional blanks do not independently prove either role.

This allows batch, purity, and impurity measurements to span the same detail run when source evidence supports them, while test number and test type remain row-level cells.

### Geometry and Lineage Validation

Blank-cell runs are necessary but not sufficient. At least one of these source-backed validations is required:

- source cell boundary covers the candidate y-range;
- anchor word bbox occupies the group region and no competing anchor occurs before the next group;
- semantic row lineage proves that the detail rows were split from words within the same source group region;
- a validated early physical row group maps completely onto the final semantic run.

If final provenance is absent or contradictory, the resolver must fail closed and leave the table unspanned.

## Integration With Semantic Projectors

Semantic projectors that rebuild a grid must also rebuild row provenance. The grouped study-metric projector will return projected rows with their contributing word references and y-ranges instead of returning text rows alone.

The resolver is table-family independent. Projectors may provide column-role hints, but those hints are evidence rather than span instructions.

When a projector replaces `semantic_grid`, stale `presentation_spans`, canonical body spans, and coordinate-dependent row groups are invalidated before final resolution.

## Rendering

The renderer consumes canonical body spans generated from the final semantic grid. It must not infer additional rowspan from repeated values or blank cells.

For the page 94 table, the normalized logical representation produces first-column rowspans of 3, 5, 5, 2, and 7. Source-backed static columns may receive the same group extents.

## Failure Behavior

- Ambiguous group boundaries produce no span and add a review diagnostic.
- A continuation row with no detail content terminates the group.
- A nonempty competing key terminates the current group.
- Note, subtotal, and header rows are excluded from body grouping.
- Stale spans whose coordinate space no longer matches the final grid are discarded.
- Identical adjacent anchor text remains separate when source geometry proves separate groups.

## Strong-Generalization Invariants

1. No logic depends on page, section, table ID, batch code, study number, title, or expected row count.
2. Final semantic coordinates are the only coordinates used by canonical body spans.
3. Every span covers a complete group through the next proven key boundary.
4. Blank cells alone never create a canonical span.
5. Semantic grid replacement invalidates coordinate-dependent span metadata.
6. Source provenance survives physical-row splitting and semantic-row expansion.
7. Static-column spans and detail-column rows are inferred from behavior and evidence.
8. Ambiguous tables retain their content without invented structure.
9. The same resolver supports borderless and ruled tables when equivalent evidence exists.
10. Renderer output cannot contain a partial rowspan followed by blank cells belonging to the same proven group.

## Tests

### Unit Tests

- resolve variable group sizes from a final semantic grid;
- infer multiple static prefix columns and detail columns;
- reject runs containing a blank detail row;
- reject runs without provenance or geometry;
- preserve separate adjacent equal-text groups using lineage;
- invalidate stale body spans after semantic-grid replacement;
- map packed physical rows to multiple ordered semantic rows;
- terminate groups at notes, subtotals, and repeated headers.

### Page 94 Regression

- assert first-column canonical spans `LN125=3`, `94NA103=5`, `95NA215=5`, `95NB003=2`, and `96NB101=7`;
- assert each span ends immediately before the next batch anchor;
- assert trial numbers and trial types remain aligned one record per row;
- assert no blank first-column cell remains inside an accepted group in rendered HTML;
- remove the obsolete assertion that encodes `94NA103 rowspan=3`.

### Cross-Document Regression

- retain all existing valid body spans on pages 85, 102-105, and related grouped tables;
- retain negative controls where sparse blanks are not source-backed spans;
- run the complete protected r2 suite;
- run deterministic Markdown/export tests;
- audit every canonical span for bounds, overlap, coordinate-space consistency, and provenance;
- sync and hash-check the mirror before focused mirror verification.

## Non-Goals

- Reconstructing ruled borders that do not exist in the source.
- Hardcoding known batch or study identifiers.
- Treating every sparse first column as a rowspan table.
- Replacing the normalized logical record rows with source physical rows.
- Allowing individual renderers or table-family projectors to maintain separate span authorities.

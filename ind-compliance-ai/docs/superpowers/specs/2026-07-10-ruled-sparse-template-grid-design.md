# Ruled Sparse Template Grid Design

## Scope

Generalize blank, multi-column CTD template recognition for three evidence shapes represented by r2 pages 45, 49, and 52. The production implementation must not branch on filename, page number, section title, target header text, or absolute page coordinates.

## Invariants

- Keep blank forms as `structure_template`; never promote them to `business_table`.
- Preserve `row_texts`, owned block IDs, page-word text, bboxes, and rule geometry.
- Add semantic projection data without replacing source evidence.
- Suppress Markdown source rows only when their complete normalized text is represented by projected fragments.
- Prefer strong geometric evidence and reject ambiguous layouts.

## Architecture

Add a unified sparse ruled-layout analyzer under the existing `ruled_multilevel_template_header_projection` protocol. It runs after the stronger page-41 body-lattice detector and before the legacy two-band detector.

The analyzer first excludes rule bands overlapping the template title region, then selects a dominant leaf band using slot count, mapped-word coverage, horizontal order, and template containment. It returns one of three mutually exclusive `column_layout_mode` values:

1. `flat_leaf_slots`: one dominant band with at least five nonempty, distinguishable columns. This covers page 52.
2. `centered_parent_anchor_over_leaf_slots`: a parent text anchor above the leaf band is centered over two or more adjacent leaf slots even when its underline is only text width. This covers page 49.
3. `blank_group_slots_plus_repeated_leaf_anchors`: an upper band provides blank group slots and the lower leaf labels form a stable repeated period where `leaf_count == group_count * period`. This covers page 45.

## Evidence Rules

### Title exclusion

Use the title source node bbox when available. Otherwise derive a conservative title-bottom boundary from the template title bbox/first heading-like owned row. A rule band overlapping this region cannot be a table parent band.

### Flat leaf slots

- At least five leaf slots.
- Every slot maps to nonempty header text.
- Normalized headers are distinguishable.
- No accepted parent anchor or repeated group structure.

### Centered parent anchor

- The parent word is vertically above and close to the leaf band.
- Its center lies inside the union of adjacent child slots.
- Candidate children are contiguous and the parent center is close to their union center.
- At least two child slots are covered.
- Parent text must be distinct from child text.
- A title word or unrelated side note cannot qualify.

### Blank groups with repeated leaves

- At least two upper group slots and at least two leaf labels per group.
- Leaf labels have an exact normalized period repeated for every group.
- `leaf_count == group_count * leaf_count_per_group`.
- Upper group-slot centers align with the corresponding repeated-leaf group ranges.
- No visible group label is fabricated. Group indexes and colspans are semantic metadata only.

## Semantic Output

All modes emit logical columns, semantic grid, column slots, header fragments, source block IDs, confidence, and layout mode.

- Page 52 emits five flat columns and its blank body rows.
- Page 49 emits ten columns, a parent span over the last two leaf columns, and the seven toxicology-type body rows.
- Page 45 emits a leading time/stub column plus twelve repeated result columns, four blank group spans, and the `0-T h` template row. Metadata labels remain visible form rows unless represented losslessly in a composite grid.

Body fragments retain `source=page_word`, column index, and bbox. Footnote markers remain source evidence and are mapped only when their column ownership is unambiguous.

## Markdown

Reuse the structure-template semantic-grid renderer. Parent spans repeat their visible label across covered Markdown cells. Blank group spans remain blank rather than receiving synthetic `Group 1` labels. Exact consumed source rows are removed from the bullet surface; unrelated rows and notes remain visible.

## Tests

- End-to-end RED/GREEN tests for pages 45, 49, and 52.
- Synthetic geometry tests for title-rule exclusion, centered-parent acceptance/rejection, repeated-period acceptance/rejection, and group alignment.
- Focused regression protection for pages 35, 36, 37, 41, 53, and 54.
- A full r2 projection inventory to detect unexpected new triggers.

## Regression Policy

Run focused r2 tests and inventory first. Expand to complete r2/eCTD/A-tst/benchmark suites only if existing projection modes change unexpectedly, new pages trigger without strong evidence, template/business ownership changes, or focused tests fail.

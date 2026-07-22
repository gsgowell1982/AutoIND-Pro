# Stable Table Row Lineage and Context Ownership Design

## Problem

The r2 page-106 carcinogenicity result panel is parsed without evidence loss. Its
canonical semantic grid contains both the `毒代动力学：` structural row and the
following `第28 天AUC(µg-h/mla)` data row. IND-review Markdown nevertheless renders
the structural label twice and omits the AUC row.

The failure is caused by an invalid cross-layer coordinate assumption. Parser
`merged_rows` metadata is created against `display_grid`, while the Markdown
projection currently interprets its numeric `row` as a `semantic_grid` coordinate.
The dose-response projection collapses the source dose/sex header and later restores
it for presentation, so the same integer identifies different rows in the two grids.
The structural-row metadata is consequently attached to the AUC row, which is then
consumed as though it were a section label.

The same page also receives study-context rows from an absorbed structure template.
Those rows combine facts already present as ordinary study-context blocks. Exact
whole-row deduplication cannot recognize this coverage, so the same facts are
rendered twice in different row groupings.

## Goals

- Preserve every source-backed body row through semantic and presentation projection.
- Render a source structural row exactly once without suppressing its neighboring data.
- Give structural rows, presentation spans, and continuation composition one shared
  provenance contract.
- Transfer absorbed study context only when it contributes facts the owner table does
  not already contain.
- Keep all decisions evidence-backed and auditable without page, table, title, label,
  value, or coordinate special cases.

## Non-Goals

- Changing the canonical values or column schema of dose-response tables.
- Inferring new facts that are not supported by source evidence.
- Reformatting unrelated Markdown tables or structure templates.
- Removing legacy numeric row fields before all consumers have migrated.
- Fixing the source text `µg-h/mla`; it is preserved as extracted evidence.

## Considered Approaches

### Numeric row rebasing

Calculate offsets when headers are removed or inserted. This is small but fragile:
multiple transformations, duplicate header rows, continuation suppression, and future
projections can each change the offset independently. It preserves the ambiguous row
contract and is rejected.

### Normalized row-signature matching

Map metadata by normalized row content. This is useful as a compatibility path, but
it cannot uniquely identify repeated rows and can be affected by display escaping or
semantic normalization. It is retained only as a guarded fallback.

### Stable row lineage

Assign stable source-row references and propagate them explicitly through every grid
projection. This makes coordinate domains visible, supports one-to-many header
folding, and lets consumers distinguish source-backed rows from synthesized rows.
This is the selected approach.

## Row Lineage Contract

### Source rows

After `display_grid` reaches its stable postprocessed form, each row receives a
deterministic reference scoped to the table, for example:

```text
tbl_051:display_row:3
```

The table stores an aligned `display_row_provenance` list. Each entry records:

- `row_ref`: stable source-row identity;
- `source_grid`: `display_grid`;
- `source_row_number`: one-based compatibility coordinate;
- `signature`: normalized row signature used for diagnostics and legacy fallback.

`merged_rows` continues to expose its numeric `row` for compatibility, but also
records `source_grid` and `source_row_ref`. New consumers must use the reference.

### Semantic rows

`semantic_row_provenance` is aligned one-to-one with `semantic_grid`. Each entry
contains `source_row_refs`:

- an unchanged body row references its display row;
- a folded semantic header may reference both the dose and sex display rows;
- a recovered row may reference all evidence rows that contributed to it;
- a row without a defensible source mapping has an empty reference list and an
  explicit derivation reason.

Semantic projection must preserve provenance when it filters, joins, or reorders
rows. Coverage audits compare source-backed semantic body rows with the final
semantic body rows independently of presentation.

### Presentation rows

Markdown projection returns a presentation surface containing both `rows` and an
aligned `row_provenance` list. A presentation-only header row has no source row and
records its derivation. Rows copied from `semantic_grid` propagate their semantic
source references.

Structural metadata resolves to a presentation row by matching `source_row_ref`
membership. Numeric row equality is never used across different grid domains.
`presentation_spans` and continued-table composition use the same resolver.

### Legacy fallback

Tables created before row provenance is available may use normalized signature
matching only when the match is unique. If there is no match or more than one match,
the renderer preserves the candidate data row and records an unresolved-provenance
diagnostic. It must not silently consume a row.

## Markdown Rendering Behavior

The renderer treats a resolved structural row as a presentation boundary:

1. flush preceding tabular rows;
2. render the structural label once;
3. consume only the presentation row carrying the same source reference;
4. continue with the immediately following data row.

For r2 page 106, the structural reference resolves to the `毒代动力学：` row. The
AUC row has its own distinct source reference and remains in the following table
segment. Synthetic dose and sex header rows do not acquire structural identities.

## Study-Context Ownership Contract

### Typed facts

Study-context blocks expose normalized facts alongside their display text. A fact
contains:

- normalized field label;
- normalized value;
- source row or block reference;
- source owner and page;
- display text reference.

Existing context extraction remains the evidence owner. Template absorption does not
reparse arbitrary prose when typed facts are already available; it consumes the same
established label/value extraction used for study metadata.

### Coverage-based transfer

Before transferring an absorbed template row, compare its fact set with the owner
table's existing fact set:

- `missing`: transfer the missing facts with their evidence references;
- `already_covered`: retain provenance on the absorbed template but do not append a
  visible context row;
- `ambiguous`: retain the evidence and diagnostic without presenting a duplicate.

A combined template row is fully covered when every normalized fact it contains is
already owned by the table, even if the table presents those facts across multiple
rows. Partial coverage transfers only missing facts; it does not repeat covered facts
inside the original combined string.

### Audit

The owner table records a transfer audit containing source template id, candidate fact
count, transferred fact count, already-covered fact count, ambiguous fact count, and
the resulting visible context block references. The absorbed template remains
metadata-only after ownership transfer.

## Components and Boundaries

- PDF table postprocessing owns stable display-row references and attaches them to
  `merged_rows`.
- Typed semantic adapters own `semantic_row_provenance` and row-coverage audits.
- The Markdown adapter owns presentation-only row synthesis and provenance
  propagation, but cannot invent source identities.
- Structure-template absorption owns fact-coverage comparison and transfer audits.
- Rendering consumes typed ownership and provenance results; it does not infer them
  from page numbers or domain-specific text.

## Error Handling and Safety

- Validate that every provenance list length matches its grid length.
- Reject duplicate `row_ref` values within one source grid.
- Preserve rows when structural ownership is unresolved or ambiguous.
- Never use a page number, table id, title, exact label, exact value, or bounding-box
  threshold to select the page-106 behavior.
- Keep source evidence, raw grids, semantic grids, and absorbed template evidence
  unchanged when presentation transfer is suppressed.
- Emit diagnostics for incompatible coordinate domains and incomplete fact coverage.

## Testing Strategy

### Deterministic row-lineage tests

- A two-level source header folded to one semantic header and restored to two
  presentation rows.
- A structural row followed immediately by a data row; the label renders once and the
  data row survives.
- Duplicate semantic row signatures prove that stable references, not text order,
  control structural ownership.
- A legacy table with one unique signature exercises the compatibility fallback.
- An ambiguous legacy signature preserves all rows and emits a diagnostic.
- Presentation spans and continued-table row suppression retain their correct source
  references.

### Real-document regression tests

- r2 page 106 semantic grid contains the AUC row.
- The IND-review section contains the exact AUC values once.
- The visible `毒代动力学：` structural title occurs once and is not also a table row.
- All source-backed semantic body rows are accounted for in the presentation coverage
  audit.
- Existing page 97-100 source-sex-header behavior and page 103-104 presentation spans
  remain unchanged.

### Context-ownership tests

- A combined absorbed row fully covered by two existing atomic context rows adds no
  visible row.
- A partially covered row transfers only its missing fact.
- Repeated labels with different values remain distinct.
- r2 page 106 no longer repeats the context facts sourced from
  `structure_template_064`.
- The transfer audit accounts for every candidate fact.

## Delivery Sequence

1. Introduce row-lineage contracts and deterministic failing tests.
2. Propagate lineage through semantic and Markdown projection.
3. Migrate structural rows, presentation spans, and continuation composition.
4. Add page-106 end-to-end content-conservation regression coverage.
5. Introduce typed study-context fact coverage and transfer audit.
6. Add context ownership tests and the page-106 duplicate-context regression.
7. Run focused, cross-type, and full deterministic verification before synchronizing
   the authoritative project to its mirror.

## Success Criteria

- The page-106 AUC row is visible with values `- - 10 12 40 48 815 570`.
- The page-106 toxicokinetics structural label is rendered exactly once.
- The page-106 absorbed study-context facts are not visibly duplicated.
- No data row is consumed through an unresolved or ambiguous provenance mapping.
- Existing dose-response, continuation, semantic HTML, and source-backed span
  regressions pass.
- No production rule depends on r2, page 106, table `tbl_051`, or its literal text.

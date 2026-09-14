# Dose-Response Source Header and Row Provenance Design

## Problem

The r2 page-98 dose-response table has internal leaf identities such as `0 M`, `0 F`, `200 M`, and `200 F`, but the source page does not contain a standalone sex header row. Markdown treated the internal leaf identity as display evidence and synthesized `| Sex | M | F ... |`.

The same projection inserted a display row without rebasing source `merged_rows` coordinates. On page 99, source row 17 is the `Post-dose evaluation:` separator. After the synthetic row was inserted, row 17 in the projected grid was `Additional examinations`, so Markdown suppressed the valid data row and rendered the separator as an ordinary table row.

## Decision

Keep internal column identity and source-visible header evidence separate:

- `has_sex_leaf_columns` describes the normalized leaf schema.
- `source_has_explicit_sex_header_row` records whether an independent sex header row exists in source evidence.
- Markdown always restores validated dose parent groups, but emits a separate sex row only when the source-evidence flag is true.

Treat `merged_rows.row` as a semantic-source row coordinate. Before rendering a projected grid, derive a projected-row-to-source-row map from normalized row signatures, consuming duplicate signatures in source order. Resolve merged-row metadata through that map instead of applying source coordinates directly to the changed presentation grid.

## Boundaries

- The parser owns source evidence and normalized semantic identity.
- The Markdown adapter owns presentation-only row materialization.
- `semantic_grid` remains unchanged by display projection.
- Projected rows with no source counterpart have no merged-row identity.
- Existing source row coordinates remain a fallback only when no mapped projected row is available.

## Safety

- Do not infer source-visible rows from internal leaf labels alone.
- Do not key behavior by page, table id, study title, compound, or exact row values.
- Preserve duplicate semantic rows by matching normalized signatures in source order.
- Preserve existing behavior for tables without projection or merged-row metadata.
- Carry the source-evidence flag through both ordinary panel recovery and page-word-only continuation recovery.

## Verification

- Deterministic no-source-sex-row fixture.
- Deterministic source-backed-sex-row fixture.
- Deterministic projected-header plus merged-row coordinate fixture.
- Full deterministic Markdown export module.
- Focused real r2 page 97-100 result-panel regressions.
- Syntax checks, whitespace checks, mirror hash equality, and focused mirror tests.

# Genotoxicity Logical Parent Header Design

## Problem

The r2 page-103 chromosomal-aberration table enters postprocessing as a nine-column sparse visual grid. Early borderless-table analysis records a source-backed `cytotoxicity a` parent candidate over raw columns 3-7, but two alignment-only columns disappear when the typed genotoxicity projection creates seven logical columns. The typed projection reconstructs the correct four result leaves but does not translate the parent group into logical coordinates. Markdown therefore receives no genotoxicity span and renders one flat header row.

## Decision

Chromosomal-aberration header projection owns the raw-to-logical conversion. It preserves the canonical seven leaf columns and promotes a parent span only when source metadata proves all of the following:

- the parent candidate is `cytotoxicity a`;
- its child evidence contains `% control`, mean chromosome aberration rate, `Abs/cell`, and total polyploid cells;
- the canonical logical header contains those four result leaves contiguously.

The committed span uses logical coordinates `col=3, colspan=4` and source `genotoxicity_multilevel_header_projection`. Same-assay continuation tables inherit the validated logical spans from their parent projection.

## Boundaries

- Generic borderless analysis detects source candidates in visual coordinates.
- The assay-specific adapter validates semantic meaning and converts to logical coordinates.
- Markdown only materializes validated logical spans; it does not infer assay structure from text.
- Canonical leaf headers remain the stable continuation and data-alignment schema.

## Safety

- Do not reuse raw start/end columns directly after sparse-column compaction.
- Do not synthesize a parent span from the canonical header alone.
- Require the complete four-leaf evidence set before promotion.
- Inherit spans only when parent and child have the same genotoxicity assay kind.
- Preserve existing bacterial reverse-mutation, DNA-repair, and micronucleus behavior.

## Verification

- Real r2 page-103 AST must contain `cytotoxicity a`, logical `col=3`, and `colspan=4`.
- Page-103 Markdown must render the parent row followed by the four canonical leaves.
- Page-104 continuation must inherit the same logical span.
- Pages 102-105 genotoxicity regression and deterministic Markdown projection must remain green.

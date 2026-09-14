# Evidence-Complete Study Panel Transfer Design

## Goal

Preserve a visual study panel's title, metadata fields, object prelude, result table, and table-note groups when the detector initially places all surfaces in one visual grid.

## Problem

The current refinement can build a text-complete metadata template from synthetic visual cells that have no bbox or source block identity. That non-null template prevents the word-evidence fallback, while the source table is still cropped. The template then cannot be inserted into the page AST, so the transferred title and fields disappear.

Markdown has a separate failure: document-wide note deduplication uses normalized text as identity. A generic group label such as `附加信息：` can legitimately occur under two different tables, so choosing one preferred table owner deletes valid content.

## Architecture

### Evidence-complete transfer

A visual study metadata transfer is committable only when the destination template has:

- a non-empty title and metadata rows;
- a valid bbox derived from source evidence;
- a source table identity;
- populated fields or sections consistent with the detected panel.

Synthetic cell rows remain useful for text recovery, but they are not authoritative geometry. When their row bboxes are unavailable, the parser must reconstruct matching visual rows from the table's word evidence. The source table is cropped only after a committable destination template exists. If neither source can build one, the source grid remains intact and receives a diagnostic rather than losing content.

### Source-aware row recovery

Word evidence is clustered into physical rows. Each recovered row carries its display text, bbox, source table ID, and stable table-row reference. This evidence is used for metadata-template bbox construction, field bbox projection, and leading object-prelude geometry.

### Note-group identity

Document-wide preferred-owner deduplication applies only to genuinely owner-ambiguous note copies. A segment with `note_group_id`, or a pure group-boundary label such as `附加信息：`, is allowed to repeat across tables. Exact duplicates inside one table are still removed by the existing local segment deduplication.

## Data Flow

1. Detect the study section and result-matrix boundary in the visual grid.
2. Attempt metadata construction from grid rows plus cell geometry.
3. Reject that candidate when transfer evidence is incomplete.
4. Rebuild the same metadata rows from word geometry.
5. Validate the destination template.
6. Attach preludes and note-reference edges.
7. Crop the source table and synchronize template/table objects into page AST.
8. Render each table's note groups independently, preserving group order.

## Regression Protection

- Real r2 page 85: title and metadata template exist in page AST with bbox.
- Real r2 page 85: four-column study-condition projection is present.
- Real r2 pages 84-85: two legitimate `附加信息：` labels both render in their owning positions.
- Real r2 pages 85-86: the page-85 note group renders label, starred route note, and `n.d.` definition before section 2.6.5.13.
- Synthetic Markdown contract: same normalized label under different owners is not globally removed.

## Scope

Changes are limited to visual study-panel transfer in `parsers/pdf/postprocess.py`, note rendering normalization in `api/main.py`, and focused parser/Markdown tests. No filename, page number, compound name, coordinate, or sample-specific production rule is introduced.

# Source-Aware Study-Panel Parent Header Design

## Goal

Preserve table parent headers that are physically located between populated study metadata and a borderless business table, without merging them into the preceding metadata field.

## Architecture

The table-adjacent study-panel builder classifies terminal physical rows before metadata continuation merging. At this early stage a table may still have a packed grid, so a strong measurement-header profile can preserve the row as an independent typed candidate without guessing its final colspan; word-backed leaf anchors provide additional support when already available. The late pre-table projection runs after table normalization, validates the final header structure, transfers ownership to the table, and writes `pre_table_parent_header_projection`.

Metadata row construction emits source-aware row records. Each logical row carries its text, bbox, and all contributing `source_block_ids`, so valid wrapped metadata values can merge without breaking later ownership resolution. Template parent-header release consumes these row records when available and retains the positional fallback for older template producers.

## Data Flow

1. Inspect the current table header and derive word-backed leaf anchors when the early grid supports them.
2. Scan only the terminal rows after the last key-value metadata field.
3. Preserve a close terminal row as a separate candidate when it has strong measurement-header semantics or verified multi-column support.
4. Build metadata fields independently from typed parent-header candidates and retain one-to-many source provenance.
5. Let the late normalized-table pass validate and transfer candidates to the business table.
6. Render the semantic two-row header through the existing Markdown projection.

## Safety

- Do not use page numbers, document names, exact study titles, or exact header text.
- Require table adjacency, word-backed leaf anchors, a multi-column range, and non-prose header text.
- Keep genuine wrapped metadata continuations in metadata and preserve all of their source IDs.
- If geometry is incomplete, retain the existing ownership instead of forcing a table header.

## Verification

- Synthetic contract for geometry-based terminal-row release.
- Synthetic contract for one logical metadata row with multiple source blocks.
- Real r2 page 81 parent header spanning six columns.
- Existing page 82 multi-parent spans and page 83 adjacent header recovery.
- Focused Markdown export and mirror verification only, per the risk-based regression policy.

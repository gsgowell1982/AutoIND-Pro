# Source-Faithful Table Cell Spans Design

## Goal

Represent and render true table cell rowspans and colspans consistently for real IND documents, while distinguishing merged cells from multiline text inside an ordinary cell. The immediate evidence set is r2.pdf sections 2.6.7.8A, 2.6.7.8B, and 2.6.7.9A on PDF pages 102-104.

## Current Failure

The parser and renderer use separate span surfaces:

- `span_header_cells` for selected semantic header projections;
- `presentation_spans` for selected body row groups;
- `logical_cells` containing mostly unit cells plus some overlapping span cells;
- renderer-side inference and continued-table span extension.

This separation causes contradictory results. Section 2.6.7.8A omits the two-row spans for `代谢活化` and `供试品`, so the HTML repeats them in the leaf-header row. Section 2.6.7.8B mistakes multiline text in `细胞毒性a (%对照)` for a four-column parent header. Cross-page `MM-180801` body groups in 2.6.7.8A and 2.6.7.8B are not consistently merged. Section 2.6.7.9A is flattened to one header row even though the neighboring tables take different presentation paths.

## Canonical Span Contract

Every business table may expose `cell_spans`, an ordered list of canonical span records:

```json
{
  "span_id": "tbl_recovered_102_01:cell_span:1",
  "role": "header",
  "coordinate_space": "semantic_grid",
  "row": 0,
  "col": 3,
  "rowspan": 1,
  "colspan": 5,
  "text": "实验#1 回复突变体菌落计数(平均值±SD)",
  "source_pages": [102],
  "source_table_ids": ["tbl_recovered_102_01"],
  "source_cell_refs": [],
  "span_group_id": null,
  "evidence": "source_header_geometry",
  "confidence": 0.96
}
```

Required invariants:

- coordinates refer to `semantic_grid` unless explicitly marked as `logical_table_chain`;
- `rowspan` and `colspan` are positive integers and at least one is greater than one;
- the anchor text matches the semantic cell at `(row, col)`;
- covered cells contain no conflicting non-empty value;
- every record retains source page/table lineage and evidence type;
- overlapping spans are rejected unless they are linked physical fragments of one cross-page `span_group_id`.

`span_header_cells` and `presentation_spans` remain temporarily available as compatibility adapters. New rendering and audits use `cell_spans` as the authoritative contract.

## Header Interpretation

### True multilevel header

A header colspan requires source evidence that one parent label governs multiple leaf columns. Accepted evidence includes a parent label centered over multiple leaf anchors, a source horizontal rule covering those children, or an existing high-confidence parent/child header group with consistent geometry.

Trailing blank cells alone do not prove colspan.

### Multiline ordinary header

Fragments aligned to one column, such as `细胞毒性a` plus `(%对照)`, are fused into one semantic header cell. They do not become a parent header. Presentation may use a line break inside the cell, but the AST records no colspan.

### Header rowspan

In a true two-level header, standalone columns that have no leaf child span both header rows. This applies to `代谢活化`, `供试品`, and `剂量水平(µg/皿)` in 2.6.7.8A.

## Body Row Groups

Body spans are derived from source row lineage and group boundaries, not from globally unique text matching. Repeated labels such as `MM-180801` are resolved using source row references, physical order, column, and contiguous group extent.

For a table split across pages:

- each physical segment retains local span evidence;
- matching terminal and leading fragments receive the same `span_group_id`;
- the chain owner exposes a `logical_table_chain` span with combined global rowspan, source pages, and source table IDs;
- the renderer consumes the global span when rendering a merged continued-table chain.

## Target Outcomes

### 2.6.7.8A

- header `(0,0)`, `(0,1)`, and `(0,2)` each have `rowspan=2`;
- header `(0,3)` has `colspan=5` and the five strain leaf headers remain visible;
- `无代谢活化` and its first `MM-180801` group preserve their source-backed rowspans;
- the `有代谢活化` and `MM-180801` groups continue across pages 102-103 without repeated cells.

### 2.6.7.8B

- the seven semantic headers are independent cells;
- there is no `细胞毒性a` parent colspan;
- multiline label/unit fragments remain inside their own headers;
- both activation groups and both `MM-180801` groups render with correct rowspans, including the page 103-104 continuation.

### 2.6.7.9A

- the five headers remain independent cells with source-faithful line wrapping;
- `MM-180801` has `rowspan=4`;
- the continuation positive control remains a separate row.

## Rendering

Semantic HTML is used whenever canonical spans are present because pipe-table Markdown cannot represent merged cells. The renderer must not infer a colspan from blank semantic cells when a canonical span contract exists. It validates canonical spans, renders each anchor once, suppresses covered cells, and preserves multiline header text without inventing parent groups.

## Compatibility

Existing table families continue to work through adapters:

- high-confidence legacy `span_header_cells` become canonical header spans;
- source-backed legacy `presentation_spans` become canonical body spans;
- renderer fallback inference remains available only for tables without `cell_spans` during migration.

No raw, display, or semantic grid is overwritten. Span projection is a presentation/structure layer with explicit provenance.

## Testing

Tests must fail before implementation and cover:

- true colspan versus multiline-cell negative cases;
- standalone header columns spanning a real multilevel header;
- repeated body labels resolved by source lineage rather than text uniqueness;
- cross-page body span linking and logical-chain rowspan;
- AST span coordinates, source pages, source table IDs, and non-overlap invariants;
- exact HTML rowspan/colspan for 2.6.7.8A, 2.6.7.8B, and 2.6.7.9A;
- preservation of all data rows, positive controls, continuation rows, notes, and existing protected r2 behavior.

## Scope

Changes are limited to table semantic/span projection, continued-table chain merging, semantic HTML span consumption, and focused parser/export regression tests. No page-number special cases, PDF mutation, or Markdown text cleanup is permitted.

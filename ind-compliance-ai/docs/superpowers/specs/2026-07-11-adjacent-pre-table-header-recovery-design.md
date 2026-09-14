# Adjacent Pre-Table Header Recovery Design

## Goal

Restore source-faithful IND table context and multilevel headers when labels are physically above a borderless business table rather than inside its detected grid.

## Architecture

Use one generic adjacent-pre-table evidence pass after borderless tables are available. The pass consumes nearby unowned text blocks and classifies them by geometry and table-header semantics:

- `compact_study_context_field`: two or more vertically adjacent colon fields immediately before a table; preserve each physical row as a separate Markdown list item.
- `wrapped_leaf_header`: a short fragment aligned with one leaf column and directly stacked above its detected leaf text; combine the two source fragments into one logical column label.
- `parent_span_header`: a centered label whose horizontal footprint covers at least two adjacent leaf columns; repeat it across those columns in the semantic parent row and emit `span_header_cells`.

Existing study-metadata-based parent header recovery remains valid. The new pass handles pages where no `study_metadata` template was created and writes the same `pre_table_parent_header_projection` contract so downstream consumers remain unified.

## Data Flow

1. Collect text blocks on the same page ending within 40 points above a business table.
2. Exclude page furniture, section titles, notes, and already owned table/template evidence.
3. Derive leaf-column x anchors from page words in the detected header row.
4. Merge a single-column prefix with the aligned leaf label when the combined text is a coherent header and the prefix does not span multiple anchors.
5. Map wider centered labels to adjacent leaf columns and build the semantic two-row header.
6. Mark consumed source blocks as table-owned so they cannot leak into body Markdown.
7. Mark compact study-context fields as `body_list_item` with `unmarked_indented` style.
8. Export semantic grids whenever `pre_table_parent_header_projection` exists.

## Page Expectations

- Page 81: `浓度(µg/ml)` spans six columns and is visible in Markdown.
- Page 82: `Ct` spans two columns and `最后一个时间点` spans three columns in Markdown.
- Page 83: render two bullets for `研究系统：体外` and `靶向实体、试验系统和方法：血浆、超滤法`; combine `试验` plus `编号` into `试验编号`; span `CTD 中的位置` across `卷` and `页码`.

## Safety

Require close vertical adjacency, valid table overlap, distinct leaf anchors, short non-prose labels, and source ownership transfer. Do not use page numbers, absolute coordinates, exact study titles, or fixed trial values in production rules.

## Testing

Add synthetic tests for compact context rows, wrapped leaf headers, parent spans, and Markdown semantic-grid selection. Add real page 81-83 assertions for AST ownership and final Markdown. Run focused page 80-86 and multilevel-header regressions.

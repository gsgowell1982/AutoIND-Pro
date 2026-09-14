# Cross-Page Table Note Physical Order Design

## Problem

The r2 page-97 result table owns two local note rows, followed by a definition note on page 98. Parser ownership preserved that order, but Markdown assigned physical-order rank only to selected note sources. The cross-page note received rank 20 while local `dose_response_result_panel_note_row` segments received rank 21, so source class overrode page order. The renderer then joined the reordered segments into one paragraph.

## Decision

For table notes with valid source geometry, physical order is source-independent. Resolve the physical page from the canonical occurrence fields (`physical_page`, `continuation_page`, `continued_on_page`, then `page`) and sort by page, y, x, and original index. Source types continue to control ownership, deduplication, and survival, but not order.

Distinct table-note segments spanning multiple physical pages render as separate lines. Single-page multi-segment notes retain the existing paragraph/group behavior unless an explicit `note_group_id` and `presentation_mode=lines` contract applies.

## Safety

- Preserve explicit note-group line ordering.
- Require both valid bbox and positive physical page before using physical sorting.
- Fall back to existing relation/source ordering when physical evidence is incomplete.
- Do not add page, table, title, note-text, or source-family exceptions.

## Verification

- Deterministic mixed local/cross-page note fixture.
- Existing explicit note-group fixture.
- Real r2 page-97 exact line order.
- Focused r2 protection for pages 85, 87, 88, 102-104, and 111-116.


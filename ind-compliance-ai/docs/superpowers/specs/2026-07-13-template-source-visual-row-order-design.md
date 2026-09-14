# Template Source Visual Row Order Design

## Goal

Preserve physical source order when sparse rows are attached to a template after its initial construction, including r2.pdf page 65 where `日剂量(mg/kg) 0(对照)` must be the first bullet.

## Model

`inline_form_row_projection` carries `source_visual_rows`, with one record per owned source instance:

- `source_block_id`
- `text`
- `bbox`
- `page`
- physical row order

The records are complete only when their source-instance signature multiset covers the template's renderable `row_texts`. Late ownership attachment remains allowed, but physical records become the ordering authority instead of append order.

## Projection

Projected inline rows retain `source_block_ids`. Markdown inserts a projected row at the earliest consumed source instance, skips the remaining consumed instances, and emits all unconsumed source rows in physical order. Source IDs, not text signatures, distinguish duplicate text instances.

`row_texts`, fields, and sections remain available for compatibility. When source visual row coverage is incomplete, Markdown uses the existing signature-based fallback.

## Regression Protection

- Page 65 dose/control row is the first visual bullet and precedes female toxicokinetics.
- Projected rows render once and source cells do not render separately.
- Duplicate text instances are consumed by source ID rather than by the first matching string.
- Pages 64 and 67-69 retain their established inline-row behavior.

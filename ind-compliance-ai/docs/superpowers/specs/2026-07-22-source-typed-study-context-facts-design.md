# Source-Typed Study Context Facts Design

## Problem

Study context is currently rendered as row text and later reparsed into `(label, value)` facts for ownership comparison. This loses field boundaries that were available at extraction time.

On r2 page 114, the source row contains three visually separate fields:

- `首次给药日期：1995 年10 月8 日`
- `剔除/未剔除的仔鼠：淘汰到4 只/性别/窝`
- `GLP 依从性：是`

The late text parser does not recognize the middle label. It therefore includes the middle field in the first field's value. An absorbed template already contains the correct standalone date field, so exact fact comparison treats it as missing and appends a duplicate date after `F1 雌性：75 mg/kg`.

The defect is not Markdown duplication. It is loss of typed field ownership before transfer.

## Decision

Introduce a canonical source-typed fact contract on `study_context_blocks` and migrate the two producers that already have authoritative source evidence:

1. Dose-response panel context built from positioned PDF word rows.
2. Absorbed structure-template context built from typed template `fields`.

All other producers retain the existing text parser as a compatibility fallback. The fallback cannot replace facts supplied by an authoritative producer.

## Canonical Fact Contract

Each fact has this shape:

```python
{
    "label": str,
    "value": str,
    "fact_key": str,
    "display_text": str,
    "source_ref": str,
    "source_owner": str,
    "source_kind": "source_geometry" | "template_field" | "text_fallback",
    "page": int,
    "bbox": list[float],
}
```

Required invariants:

- `label`, `value`, and `fact_key` are non-empty for ownership facts.
- `fact_key` is derived only from normalized label and normalized value.
- Same label with different values remains distinct.
- Facts preserve source owner and source kind.
- Display text is not used as identity.
- Producer-supplied facts are never reparsed or overwritten by fallback logic.

## Dose-Response Producer

`_dose_response_result_panel_from_page_words` already has ordered word items and bounding boxes for every metadata row. Before joining a row for display, it will split the row into visual field clusters.

Field-cluster rules:

- Keep physical x order.
- Estimate normal within-field spacing from positive adjacent word gaps.
- Split only on a materially larger gap and only when the next cluster contains a colon-bearing label/value fragment.
- Parse each cluster independently at its first colon.
- Preserve the cluster bbox and source fragment texts.
- If geometry cannot establish more than one valid field, use the existing row parser as fallback.

For the page-114 source row, this yields three facts before the display row is joined. The first date value ends at `日`; it cannot absorb the culling field.

The resulting context block retains its existing `text` for Markdown and gains aligned `study_context_facts` with `source_kind="source_geometry"`.

## Template Producer

Absorbed structure templates already expose `fields` containing `text`, `label`, and `value`. Transfer will build candidate fact rows from these fields instead of reparsing `original_rows`.

Rules:

- A populated field with non-empty label and value becomes one `template_field` fact.
- Preserve template field order.
- Associate a field with the matching original display row by normalized text and occurrence order.
- Rows without a typed populated field may use the existing text fallback.
- Blank labels such as `未见不良反应剂量：` remain presentation context, not ownership facts.

This makes `剔除/未剔除的仔鼠` valid without adding it to a global literal-label list.

## Compatibility Fallback

`_ensure_study_context_fact_records` becomes fill-only:

```text
valid producer facts present -> preserve them
facts absent                -> parse block text as text_fallback
invalid producer facts      -> retain audit evidence and use fallback
```

The existing label grammar remains for legacy context producers. It is no longer the authoritative source for migrated producers.

## Transfer And Audit

Absorbed context transfer continues to compare exact `fact_key` values.

For each candidate row:

- Remove facts already owned by the destination.
- Transfer only missing facts.
- Preserve the original row text when every fact in that row is missing.
- Reconstruct display text from missing facts when coverage is partial.
- Do not emit a block when every candidate fact is covered.

The transfer audit adds counts by source kind and records whether destination coverage came from authoritative facts or fallback facts. Existing audit keys remain compatible.

For page 114, the standalone date candidate matches the authoritative geometry fact and produces zero transferred facts.

## Failure Handling

- Invalid or overlapping geometry clusters do not authorize fact ownership.
- Empty field values do not create fact keys.
- Ambiguous template rows remain in `ambiguous_rows` and are not transferred as facts.
- Producer facts with duplicate keys are consumed once in source order.
- No page, section title, study number, field value, or fixed x-coordinate rule is permitted.

## Testing

Focused unit tests will prove:

1. A positioned three-field row produces separate date, culling, and GLP facts.
2. Fill-only enrichment preserves authoritative facts.
3. Template fields are used even when the fallback label grammar does not recognize the label.
4. A fully covered template candidate transfers no visible block.
5. Same label with a different value still transfers.

Real r2 regression will prove:

- Page 114 renders the first-dose date exactly once.
- No absorbed context bullet appears after `F1 雌性：75 mg/kg`.
- The culling and GLP fields remain visible in their original combined line.
- Page 106 partial fact transfer and page 103-105 genotoxicity context remain intact.

## Scope

Modify:

- `parsers/pdf/postprocess.py`
- `tests/parser_tests/test_r2_regression.py`
- `D:/ind-session/ENGINEERING_DECISIONS.md`

Do not modify Markdown rendering. Do not migrate unrelated study-context producers in this iteration. Synchronize changed code and tests to the mirror after verification.

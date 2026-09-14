# Cross-Page Open Note Group Design

## Goal

Render a table-ending note label and its next-page continuation as one source-ordered note group owned by the preceding table.

## Semantic Model

A result-matrix row containing only a note label such as `附加信息：` is not part of the semantic grid. It becomes a table-owned note segment with `relation=below`, its source page, and a stable note-group role. A qualifying note at the top of the next page, before the next section boundary, remains owned by the preceding table through its existing `owner_table_id`, `logical_owner_page`, and `note_scope=previous_table` evidence.

The local label and cross-page continuation remain separate ordered segments so Markdown renders:

```text
附加信息：
a-总放射性；回收率，14C
```

## Ownership Rules

- Explicit cross-page ownership outranks same-page geometry and marker-reference heuristics.
- A top-of-page note already claimed as a previous-table continuation must not also become a `same_panel_study_metadata_note` for the following table.
- Markdown owner normalization must preserve explicit `owner_table_id` evidence and must not penalize cross-page ownership.

## Presentation Boundary

Both note segments render after the owner table and before the next page's first new section structure. The continuation is absent from the following table's note blocks and is rendered exactly once.

## Regression Scope

Protect page 86 note extraction, page 86-to-87 continuation ownership, page 87 non-ownership, and final Markdown ordering. Re-run nearby page 85-to-87 cross-page note tests and shared table-note normalization tests.

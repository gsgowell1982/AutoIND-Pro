# Generalized Study Context Facts Design

## Goal

Prevent duplicated or reordered study-context bullets when a populated structure template is absorbed by a business table, while generalizing to real IND documents with inline multi-field rows, field variants, placeholders, cross-page context, and conflicting sources.

## Current Failure

The visible page-112 text is parsed into atomic facts (`种属/品系=新西兰兔`, `剖腹产日=G29`). The absorbed template parses the same inline row as a composite value (`种属/品系=新西兰兔 剖腹产日：G29`). Exact fact-key comparison treats the composite as missing and appends it after `F1 仔畜：5 mg/kg`.

## Design

### 1. One fact parser

Main-table context, structure-template fields, and continuation context use one parser. It recognizes field boundaries from normalized punctuation, line/word spacing, known schema labels, and candidate-label scoring. Known labels are extensible; no page-specific or single-label special case is allowed. Unknown labels remain as raw low-confidence fields rather than being silently folded into a preceding value.

### 2. Atomic facts plus raw source row

Each context block retains its source display text and an ordered list of atomic facts. Fact records include normalized key, display text, source kind, physical page, bounding box, source row reference, and owner. A composite row is never represented as one long value when multiple field boundaries are recoverable.

### 3. Conservative transfer and provenance

Template absorption compares atomic fact keys. Existing source-owned facts have precedence. Only facts absent from the visible owner may transfer. Partial coverage transfers only the missing atomic facts; conflicts (same canonical label, different populated value) are recorded in an audit and are not silently appended as duplicate visible bullets. Low-confidence unsplit template rows are audit-only.

### 4. Physical ordering

Transferred facts carry source position where available. Rendered context remains in source order. If a transferred fact has no reliable source position, it is not appended to the visible body after a later source row; it is retained in provenance/audit for review.

### 5. Regression invariants

Tests cover:

- inline rows containing multiple fields, with punctuation and field-order variants;
- populated and placeholder template rows;
- source/template overlap, partial coverage, and conflicting values;
- cross-page continuation and source-order insertion;
- unknown or ambiguous labels being preserved without unsafe transfer;
- pages 106, 109, 112, and 114 as end-to-end golden regressions.

For page 112 specifically, `种属/品系` and `剖腹产日` occur once, the final context bullet is `F1 仔畜：5 mg/kg`, and the result table follows it.

## Scope

Changes are limited to the shared PDF post-processing fact/transfer path, Markdown context rendering metadata, and parser regression tests. No page-specific rendering branch or source PDF mutation is permitted.

## Success Criteria

The new unit tests fail before implementation, pass after implementation, the page-112 regression no longer emits a synthetic trailing duplicate, and the complete existing r2 regression suite remains green.

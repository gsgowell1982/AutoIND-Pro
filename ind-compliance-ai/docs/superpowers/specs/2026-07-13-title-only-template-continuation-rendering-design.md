# Title-Only Template Continuation Rendering Design

## Problem

Page 71 of `r2.pdf` ends with the source-backed heading
`2.6.7.14 (1)生殖毒性 试验编号(续)`. The parser correctly creates a
same-page continuation template whose body starts on page 72, but Markdown
rendering drops that template because it has no entries, fields, sections, or
notes on page 71.

## Decision

Treat source-backed continuation-title anchors as renderable template content,
independently of whether their body rows occur on the same page. Keep the
existing empty-template suppression for every template that lacks this explicit
evidence.

Preserve `title_source_block_id` when a top-level structure template is
projected into `document_ast.pages[].blocks`. Markdown consumes the AST page
node, so the source-ownership field must be part of the shared template-node
contract rather than recovered from a secondary semantic signal.

A title-only template is renderable only when all of these conditions hold:

1. It has a non-empty title and `title_source_block_id`.
2. It is marked `is_structure_template_continuation`.
3. Its semantic state is `pending_body_on_next_page`.

The renderer will emit the normal structure-template heading and no synthetic
rows. The linked page-top continuation body remains rendered through its own
template object.

## Alternatives Rejected

- Rendering every titled empty template would expose parser artifacts and
  previously suppressed empty skeletons.
- Moving the page-71 title onto the page-72 body template would lose physical
  source order and source-page ownership.
- Adding a page-number or title-text exception would not generalize to other IND
  continuation templates.

## Regression Protection

Add a focused unit test showing that an ordinary empty titled template remains
hidden while a source-backed pending continuation anchor renders its heading.
Extend the existing page-71 full-PDF regression to require exactly one visible
heading between the page-71 terminal note and the page-72 first body row, and to
require the AST template node to retain its title source ID.

Verification is limited to the focused renderer tests, the affected page-71
regression, the nearby continuation-template regression group, Python syntax,
and diff hygiene. A complete repository regression is not required for this
localized Markdown eligibility change.

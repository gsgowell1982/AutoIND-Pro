# Repeated Cross-Page Note Occurrence Ownership Design

## Goal

Preserve repeated note text as distinct physical occurrences and attach each page-top occurrence to the correct preceding table without changing table detection or semantic projection on earlier pages.

## Problem

The page 87 and page 88 source layers each contain `a-总放射性；回收率，14C`. The first occurrence belongs to the page 86 excretion table. The second occurrence belongs to the page 87 bile-excretion table. Text-only suppression currently expands the first occurrence to the following page and removes the second occurrence before it receives an owner.

The page 87 table is also rejected as a cross-page owner because its raw table bbox ends above a fixed page-bottom threshold even though its composite study panel contains the marker anchor `TRAa` and is the final eligible table before the page 88 note.

## Ownership Model

- A note occurrence is identified by source evidence: physical page plus source block id when available, otherwise physical page plus bbox.
- Equal normalized text on adjacent pages does not imply duplication.
- A page-top marked note may attach to the preceding page's last result-matrix table when all of the following hold:
  - the note appears before the next object or section boundary;
  - the preceding table has a compatible terminal marker anchor such as `TRAa` for marker `a`;
  - no closer eligible table has already claimed the occurrence.
- Existing near-page-bottom and continuation-state evidence remains valid but is no longer required when marker-anchor evidence is strong.

## Suppression Contract

- Suppress a source text node only after a table has claimed that exact occurrence.
- Prefer source block ids. When ids are unavailable, match the claimed physical page and bbox/text evidence.
- Never add a claimed note's normalized text to `physical_page + 1` without evidence for a continuation source instance.
- If ownership cannot be resolved, preserve the text in the document AST and mark it for review rather than deleting it.

## Presentation

The two identical notes remain independent:

- page 87 source occurrence renders once after the page 86 table and before the page 87 example;
- page 88 source occurrence renders once after the page 87 table and before the page 88 example/toxicology heading.

An explicit note label remains before its continuation and on a separate line. A label is not synthesized for a table that has no source-backed label.

## Regression Scope

- Protect both physical occurrences and their distinct owner ids/pages.
- Protect page 87 and page 88 next-object non-ownership.
- Protect final Markdown ordering and exactly-once rendering for both occurrences.
- Re-run the existing page 80, page 84-87, and shared cross-page note tests affected by marker ownership and suppression.
- Do not run the full regression suite unless a focused failure indicates broader impact.

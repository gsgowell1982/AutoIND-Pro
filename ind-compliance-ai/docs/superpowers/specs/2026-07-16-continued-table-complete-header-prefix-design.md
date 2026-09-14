# Continued Table Complete Header Prefix Design

## Problem

Continued genotoxicity tables retain their own semantic header schema. After a parent span is materialized, both root and continuation tables may have a two-row Markdown header prefix: a parent-group row followed by a leaf row. The chain merger removed at most one matching leading row. It removed the continuation parent row but appended the inherited leaf row between root data and continuation data.

## Decision

Derive the root table's complete equivalent header set from its projected grid. When merging a continuation, consume matching rows only from the continuation's leading prefix. Each root header layer may match at most once; remove the matched candidate before evaluating the next leading row.

This supports both forms:

- continuation repeats the complete parent-plus-leaf prefix;
- continuation carries only one equivalent header layer.

## Safety

- Match only exact normalized header-row equivalence.
- Consume only consecutive rows at the start of a continuation.
- Bound removal by the number of distinct root header layers.
- Never match the same root header layer twice.
- Preserve merged-row coordinate rebasing through the accumulated skipped-row count.
- Keep continuation AST headers intact; suppression is a Markdown chain-composition concern.

## Verification

- Deterministic two-level genotoxicity continuation fixture.
- Real r2 page-103 leaf header occurs exactly once.
- Pages 102-106 genotoxicity continuation and ownership regression.
- Complete deterministic Markdown export module.

# Regulation Draft Rules

Draft rule catalogs promoted from normalized regulation clauses.

## Purpose

- Keep draft regulation-to-rule mappings visible and versioned under `rules/`
  without prematurely claiming that every clause is already executable.
- Restrict this layer to dossier-checkable clauses that are strong enough to be
  considered for future implementation in the deterministic rule engine.
- Preserve exact clause citations so each future rule can trace back to a
  regulation article and the supporting material evidence contract.

## Current contents

- Direct-rule draft catalogs generated from regulation clause libraries.
- These draft rules are **not** yet wired into `build_default_material_rules()`.
- Promotion into live hard/soft rules should happen only after:
  - explicit evaluator logic is implemented
  - deterministic regressions prove the rule is stable on real materials

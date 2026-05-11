# Soft Rules

Heuristic adequacy and risk checks with explanation traces.

## Current baseline

- `SR-FACT-001`
  - Warn when comparable atomic facts disagree across document slots.
  - Consumes `material_review_contract.fact_index` and document classification rather than ad hoc UI-only aggregation.
- `SR-EVID-001`
  - Warn when detected tables or images do not project into review-ready evidence/units.
  - Consumes `material_review_contract.evidence_index` plus `unit_index` and structure references.
- `SR-FACT-002`
  - Warn when extracted atomic facts cannot be traced back to fact-eligible content units.
  - Consumes `material_review_contract.fact_index` with unit-level provenance links derived from existing content units.
- `SR-FACT-003`
  - Warn when unit-level fact signals and normalized document-level atomic facts diverge.
  - Consumes `material_review_contract.fact_signal_index` and `fact_index` instead of re-running extraction logic inside the rule layer.
- `SR-FACT-004`
  - Warn when the same document repeats a fact signal with conflicting values.
  - Consumes `material_review_contract.fact_signal_index` and stays at the objective signal layer before any downstream business interpretation.
- `SR-FACT-005`
  - Warn when extracted atomic facts are supported only by contextual roles such as table headers/titles or image nearby context.
  - Consumes `material_review_contract.fact_index` provenance and uses explicit source/role mappings rather than confidence thresholds.

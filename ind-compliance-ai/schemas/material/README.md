# Material Schemas

Unified intermediate representation schemas for parsed source materials.

## Current contracts

- `material_review_contract.schema.json`
  - Rule-ready projection over parser outputs.
  - Keeps parser-stage scope objective: structure, evidence, navigation, diagnostics, and provenance.
  - Intended as the stable handoff between parsing and downstream `rules / schemas / review` modules.
  - Document summaries now also expose auditable classification signals such as `module_label` and `quality_overview_candidate`.
  - Root indexes now include `evidence_index`, so downstream rules can audit how tables/images/TOCs were projected before fact extraction.
  - `fact_index` provenance can now carry unit-level support links when extracted facts can be traced back to fact-eligible content units.
  - `fact_signal_index` exposes unit-level fact matches before document-level normalization, so downstream rules can audit extraction alignment instead of re-scanning raw parser payloads.

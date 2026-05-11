# Phase 1 Scope

## Included

- China IND-oriented project skeleton and contracts
- Generic parser interfaces and dispatch registry (PDF/Word/PPT)
- Rule engine skeleton with hard/soft separation
- Cross-module atomic fact consistency scaffolding
- FastAPI + React (Ant Design Pro) interactive prototype
- Upload progress view, audit workbench, and consistency board

## Excluded

- Automated regulatory submission decisions
- Production-grade model training and tuning pipelines
- Full eCTD lifecycle management
- Jurisdiction-specific implementations beyond initial placeholders
- AI rule check list execution (UI placeholder only in Phase 1)

## Guardrail

Human experts remain the final authority for every filing decision.

## After Phase 1

Phase 1 establishes the deterministic parser/rule/workbench backbone. Later phases should not treat that backbone as the full product.

Planned follow-on capability lines:

- AI-assisted content consistency review
  - detect cross-document and intra-document content inconsistencies
  - surface reviewer-oriented risk prompts and evidence-linked improvement hints
  - remain advisory rather than replacing deterministic rule verdicts

- Enterprise productization
  - strengthen the reviewer UI, visual polish, and workflow ergonomics

- Grounded customer copilot
  - answer user questions with explicit grounding in parse outputs, rule results, evidence anchors, and regulation basis

- Optional RIM / CRO plugin line
  - evaluate whether parser/rule/content-consistency capabilities should also be exposed as plugin/API components
  - target enterprise integration scenarios may include:
    - RIM systems
    - submission workflow systems
    - CRO delivery pipelines
  - this line is a reserve direction, not a fixed immediate follow-on commitment

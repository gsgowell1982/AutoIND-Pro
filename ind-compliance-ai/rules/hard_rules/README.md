# Hard Rules

Deterministic existence/structure checks (pass/fail/na).

## Current baseline

- `HR-PARSE-001`
  - Parsed material must expose review-ready `content_evidence` and `content_units`.
- `HR-PARSE-002`
  - Parser review-required diagnostics must not block deterministic rule execution.
- `HR-NAV-001`
  - TOC/navigation metadata must stay internally consistent when navigation structures are present.
- `HR-CTD-001`
  - For `FIH`, explicit Module 3 / CMC file-path signals require at least one explicit quality overview / QOS document signal.

## Input contract

- These rules consume `material_review_contract.json`, not raw parser internals directly.
- Parser remains responsible for objective structure/evidence/diagnostics.
- Hard rules operate on the contract layer and may be extended by submission profile.
- CTD completeness rules should prefer auditable structural signals such as filenames and source paths before deeper semantic inference.

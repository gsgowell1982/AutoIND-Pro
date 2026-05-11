# Regulation Schemas

Structured contracts for regulation-source ingestion, clause normalization, and
rule-candidate generation.

## Current contracts

- `regulation_document.schema.json`
  - Metadata and chapter-level structure for one regulation asset.
- `regulation_clause.schema.json`
  - Clause-level normalized records with article identity, source locator, and
    initial IND-material relevance classification.
- `regulatory_rule_candidate.schema.json`
  - Candidate rule projections derived from regulation clauses before they are
    promoted into executable `rules/` logic.
- `regulatory_direct_rule_draft.schema.json`
  - Draft rule catalog for the subset of regulation clauses currently classified
    as direct dossier-checkable and ready for later evaluator implementation.
- `regulatory_requirement_matrix.schema.json`
  - Requirement-level matrix for dossier-facing regulations where one clause can
    project into one or more applicant-checkable obligations with precise
    citation anchors.

## Design intent

- Keep regulation-source ingestion separate from applicant-material parsing.
- Preserve clause-level provenance so future rule hits can cite exact articles.
- Distinguish:
  - raw/normalized regulation knowledge objects
  - executable hard/soft rules
  - requirement-matrix rows that still need reviewer or LLM-backed semantic
    grounding before becoming executable logic
  - review-only or citation-only clauses that should not become automated rules

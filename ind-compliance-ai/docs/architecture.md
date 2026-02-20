# Architecture

Phase 1 follows a controlled pipeline:

1. Upload and file-type validation
2. Parsing to normalized material schema
3. Rule evaluation (hard and soft rules)
4. Cross-module fact consistency checks
5. Risk explanation with citation anchors

## Data flow

- Input: CTD materials, supporting documents, and metadata bundles
- Parse layer: PDF/DOCX/PPTX/XML/eCTD parser registry
- Rule layer: deterministic hard rules + explainable soft rules
- Output: structured `compliance_result.json` + `audit_log.json`

## Control principles

- Determinism first for hard-rule execution.
- Citations are mandatory for regulatory references.
- LLM components are advisory, bounded, and auditable.

# Architecture

Phase 1 follows a controlled pipeline:

1. Upload and file-type validation
2. Parsing to normalized material schema
3. Preprocess progress tracking and audit workbench rendering
4. Cross-module fact consistency checks
5. Placeholder risk/rule panel for future AI checks

## Data flow

- Input: CTD materials, supporting documents, and metadata bundles
- Parse layer: PDF/DOC/DOCX/PPT/PPTX/XML parser registry
- Rule layer: deterministic hard rules + explainable soft rules
- Output: structured `compliance_result.json` + `audit_log.json`

## UI modules

- File upload and preprocess zone with live progress
- Audit workbench (PDF + bounding boxes / Markdown view / rule panel placeholder)
- Consistency board for module-level atomic fact comparison

## Control principles

- Determinism first for hard-rule execution.
- Citations are mandatory for regulatory references.
- LLM components are advisory, bounded, and auditable.

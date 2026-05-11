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

## Planned Product Layers

The long-range product plan is explicitly split into three different capability layers:

1. Deterministic compliance rules
   - hard-fail and bounded soft-risk checks for explicit eCTD / IND / regulation constraints
   - authoritative for pass/fail-style structural and metadata compliance boundaries

2. AI-assisted content consistency review
   - a separate reviewer-assistance layer for content-level checks that rules alone cannot cover well
   - target outputs include:
     - cross-document consistency findings
     - intra-document contradiction or weak-consistency risk prompts
     - evidence-linked review hints
     - bounded improvement suggestions
   - this layer is advisory and must stay explainable and evidence-anchored

3. Grounded LLM customer copilot
   - customer-facing Q&A grounded in parser outputs, rule outputs, evidence anchors, and regulation basis
   - must not be conflated with the internal reviewer-oriented content-consistency module

4. Optional pluginized enterprise integration line
   - a reserve product direction for embedding AutoIND-Pro capabilities into:
     - RIM systems
     - submission-process platforms
     - CRO workflow orchestration layers
   - should be treated as strategy-dependent rather than immediate default scope
   - later implementation should reassess:
     - integration complexity
     - deployment/security boundaries
     - product leverage versus continuing standalone reviewer-product investment

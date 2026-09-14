# eCTD Package Structure and Naming Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable enterprise-grade intake and deterministic auditing of a complete eCTD application/sequence package, starting with folder hierarchy and folder/file naming compliance while preserving the existing single-document workflow.

**Architecture:** Convert every upload mode into an immutable package inventory with normalized relative paths, file metadata, hashes, and detected application/sequence roots. Run structure and naming rules against that inventory, then expose findings through the existing job/workbench contract. XML parsing, file-format validation, cross-file consistency, risk aggregation, and human-review routing remain explicit later phases and must consume the same inventory rather than rescan the filesystem.

**Tech Stack:** FastAPI `UploadFile`, Python `pathlib`/`zipfile`/`hashlib`/XML parsing, existing rule engine and `rule_metadata.yaml`, JSON result contracts, React/TypeScript upload/workbench UI, unittest/pytest fixtures.

---

## Current Baseline

- Primary project: `D:/AutoIND-Pro/ind-compliance-ai`
- Mirror project: `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai`
- Existing upload endpoint: `api/main.py` `POST /api/v1/uploads`, currently accepts a list of `UploadFile` values.
- Existing upload validation: `api/upload_controller.py` checks individual extensions.
- Existing scope detection: `core/upload_scope_projection.py` recognizes single documents, document batches, sequence candidates, sequence packages, and application projects from relative paths.
- Existing eCTD rule inventory: `rules/rule_metadata.yaml`, including structure, naming, XML reference, checksum, schema, lifecycle, module-1, and STF rules.
- Existing eCTD rule execution: `core/rule_engine.py` and material-assessment tests in `tests/rule_tests/test_material_assessment.py`.
- Existing eCTD normalization resources: `core/ectd_region_schema_ingestion.py`, `core/ectd_module1_structure_ingestion.py`, `core/ectd_region_style_ingestion.py`, `core/ectd_stf_valid_values_ingestion.py`, and `core/ectd_controlled_vocabulary_ingestion.py`.
- Existing fixture: `tests/x202112345/0000`, useful for inventory and negative/positive structure tests. Its placeholder standard files must not be treated as an officially valid ICH/CDE package until real versioned definitions are supplied.
- Worktree is intentionally dirty. Do not reset, clean, stage all files, or overwrite unrelated changes. Run preflight before source edits and sync relevant source/tests to the mirror.

## Product Scope and Phase Boundaries

### Phase 1: package intake, hierarchy, and naming

The first production slice must support:

- ZIP upload as the primary complete-package entry point.
- Multiple-file upload with preserved `relative_path` as a compatibility path.
- Optional browser folder upload where the client supplies relative paths.
- Safe extraction and limits before parsing.
- Application root and sequence root detection.
- Immutable package inventory with normalized POSIX paths and hashes.
- Folder hierarchy rules and file/folder naming rules.
- `index.xml` and `cn-regional.xml` path coverage checks where the existing rules apply.
- Findings with rule id, severity, path, evidence, remediation, and blocking state.
- API and workbench summary showing package scope, counts, blocking failures, warnings, and human-review candidates.

### Phase 2: per-file format and content validation

Consume Phase 1 inventory to validate XML well-formedness, DTD/XSD, XSL/STF resources, PDF/DOCX integrity, file size, placeholder content, checksums, and parse status.

### Phase 3: cross-file logical consistency and review workflow

Compare index/regional XML, leaf metadata, titles, language, lifecycle, STF, study identifiers, module facts, and prior sequences. Produce evidence-linked risk items and route ambiguous or insufficient-evidence cases to human review.

Do not merge Phases 2 and 3 into the first implementation. They have different failure modes, processing costs, and evidence contracts.

## Recommended Enterprise Upload Strategy

Use three intake modes with one canonical backend inventory:

1. **ZIP package (recommended default):** preserves the submission tree, supports retry/resumable object upload, gives the server a single immutable source hash, and is suitable for audit retention.
2. **Folder upload:** support browser directory selection for convenience, but require relative-path fields and apply the same inventory builder. Treat it as equivalent to a virtual package assembled from the uploaded files.
3. **Single/multi-document upload:** retain the current flow for ordinary document review and for partial eCTD diagnostics; label it as incomplete package scope and never report package-level compliance as complete.

For ZIP intake, reject path traversal, absolute paths, duplicate normalized paths, symlink-like entries, excessive compression ratio, excessive file count, unsupported special files, and configured size limits. Store the original archive hash, extracted inventory hash, per-file SHA-256, and per-file MD5 where eCTD rules require MD5. Keep extraction in a job-scoped temporary directory with cleanup after retention policy expiry.

## Target Contracts

The package inventory should have a stable shape similar to:

```python
{
    "submission_id": "...",
    "source_kind": "zip|folder|file_batch|single_document",
    "source_archive_sha256": "...",
    "application_roots": [
        {
            "name": "x202112345",
            "relative_path": "x202112345",
            "sequences": [
                {
                    "name": "0000",
                    "relative_path": "x202112345/0000",
                    "files": [...],
                    "directories": [...],
                    "index_xml": "x202112345/0000/index.xml",
                    "regional_xml": ["x202112345/0000/m1/cn/cn-regional.xml"],
                }
            ],
        }
    ],
    "files": [
        {
            "relative_path": "...",
            "name": "...",
            "extension": ".xml",
            "size": 0,
            "sha256": "...",
            "md5": "...",
        }
    ],
    "directories": [{"relative_path": "...", "name": "..."}],
    "inventory_diagnostics": [],
}
```

Rule findings should use the existing rule result conventions and add package evidence fields where absent:

```python
{
    "rule_id": "HR-ECTD-015",
    "status": "pass|fail|warning|not_applicable|human_review|blocked_by_missing_evidence",
    "severity": "error|warning|info",
    "scope": "application|sequence|directory|file|xml_node",
    "relative_path": "x202112345/0000/m1/cn/00",
    "evidence": {...},
    "message": "...",
    "remediation": "...",
    "blocking": True,
}
```

## Implementation Tasks

### Task 1: Define the inventory contract and fixture matrix

**Status:** Initial inventory contract implemented; fixture matrix and broader intake coverage remain open.

**Files:**
- Create: `core/ectd_package_inventory.py`
- Create: `tests/deterministic_tests/test_ectd_package_inventory.py`
- Create: `tests/fixtures/ectd_packages/README.md`
- Add fixtures under: `tests/fixtures/ectd_packages/valid_minimal_sequence/`, `invalid_structure/`, and `invalid_naming/`

- [ ] Write tests for POSIX path normalization, duplicate-path rejection, application/sequence detection, per-file hashes, and preservation of empty directories where structure rules need them.
- [ ] Add positive and negative fixture manifests without copying official ICH/CDE standard files into the repository.
- [ ] Run the focused inventory tests and confirm the new tests fail before implementation.
- [ ] Implement typed inventory dataclasses/builders using `pathlib`, `hashlib`, and explicit diagnostics.
- [ ] Run the focused inventory tests until they pass.
- [ ] Commit only the inventory contract and fixtures after hunk review.

### Task 2: Add secure ZIP and folder intake normalization

**Files:**
- Modify: `api/upload_controller.py`
- Modify: `api/main.py` upload handling around `POST /api/v1/uploads`
- Modify: `core/upload_scope_projection.py`
- Create: `core/ectd_package_intake.py`
- Create: `tests/deterministic_tests/test_ectd_package_intake.py`

- [ ] Write tests for ZIP Slip, absolute paths, duplicate normalized paths, unsupported special entries, size/file-count limits, and valid nested relative paths.
- [ ] Add an intake adapter that accepts ZIP entries, folder-upload relative paths, and current file records, producing one inventory input format.
- [ ] Preserve the existing single-document path and return an explicit incomplete-package scope for it.
- [ ] Add source archive SHA-256 and per-file SHA-256/MD5 calculation without changing the parser's document file contract.
- [ ] Run intake tests and existing `test_upload_scope_projection` tests.
- [ ] Sync changed files and focused tests to the mirror, then run the same checks there.
- [ ] Commit the intake boundary after reviewing dirty-worktree overlap.

### Task 3: Implement hierarchy validation as an inventory rule adapter

**Status:** Initial adapter implemented and focused coverage passes; API integration and broader rule mapping remain open.

**Files:**
- Create: `core/ectd_structure_validation.py`
- Modify: `core/rule_engine.py`
- Modify: `rules/rule_metadata.yaml` only when a missing rule id or metadata field is demonstrated
- Create: `tests/rule_tests/test_ectd_structure_validation.py`

- [ ] Map existing rules first: `HR-ECTD-015`, `HR-ECTD-016`, `HR-ECTD-017`, `HR-ECTD-020`, `HR-ECTD-021`, and related structure rules.
- [ ] Write tests for valid `application/sequence/module/util` placement, missing required roots, wrong nesting, unreferenced directories, non-scaffold empty folders, multiple application roots, and multiple sequences.
- [ ] Implement structure checks against inventory paths and parsed index references, never against filenames/pages hardcoded in production.
- [ ] Preserve `not_applicable` or `human_review` when the package is partial or XML evidence is unavailable.
- [ ] Run focused structure tests plus directly coupled existing eCTD material-assessment tests.
- [ ] Sync source/tests to mirror and run the mirror focused set.
- [ ] Commit the hierarchy adapter after hunk review.

### Task 4: Implement naming validation as a separate rule adapter

**Status:** Initial adapter implemented and focused coverage passes; API integration and broader rule mapping remain open.

**Files:**
- Create: `core/ectd_naming_validation.py`
- Modify: `core/rule_engine.py`
- Modify: `rules/rule_metadata.yaml` only for missing metadata
- Create: `tests/rule_tests/test_ectd_naming_validation.py`

- [ ] Write tests for application identifier format, four-digit sequence names, module/region/section names, required controlled filenames, leaf character/length restrictions, forward-slash hrefs, and path traversal rejection.
- [ ] Reuse `HR-ECTD-001`, `HR-ECTD-002`, `HR-ECTD-005`, `HR-ECTD-006`, `HR-ECTD-016`, `HR-ECTD-017`, and the existing XML path rules where applicable.
- [ ] Keep placement and naming results separate so one bad path produces distinct actionable findings rather than an opaque combined error.
- [ ] Run focused naming tests and the directly coupled rule tests.
- [ ] Sync source/tests to mirror and run mirror verification.
- [ ] Commit the naming adapter after review.

### Task 5: Expose package audit results through the job API and workbench

**Files:**
- Modify: `api/main.py`
- Modify: `api/response_formatter.py`
- Modify: `ui/frontend/src/api.ts`
- Modify: `ui/frontend/src/types.ts`
- Modify: `ui/frontend/src/components/UploadPreprocessPanel.tsx`
- Modify: `ui/frontend/src/components/AuditWorkbench.tsx`
- Create or modify focused frontend tests under `ui/frontend/tests/`

- [ ] Write API contract tests for package scope, inventory counts, structure/naming summaries, blocked state, and incomplete single-document scope.
- [ ] Add package-level fields without breaking existing document job responses.
- [ ] Show application roots, sequence count, file count, structure failures, naming failures, and human-review count before expensive document parsing when possible.
- [ ] Link each finding to its relative path and evidence; preserve existing rule navigation behavior.
- [ ] Run backend focused tests and frontend unit/build checks.
- [ ] Sync relevant files to mirror and run mirror checks.
- [ ] Commit the API/UI contract after review.

### Task 6: Add end-to-end package fixtures and release gates

**Files:**
- Modify: `tests/parser_tests/test_ectd_validation_standard_regression.py` only for package-level integration coverage
- Create: `tests/cross_module_tests/test_ectd_package_audit_flow.py`
- Create: `docs/superpowers/specs/2026-09-05-ectd-package-structure-naming-audit-design.md` after design approval

- [ ] Run a valid minimal package through upload normalization, inventory, hierarchy rules, naming rules, and API serialization.
- [ ] Run invalid structure and invalid naming packages and assert deterministic rule ids, paths, statuses, and remediation evidence.
- [ ] Confirm single-document uploads remain document-scoped and cannot claim package compliance.
- [ ] Run the smallest sufficient primary regression set, then expand only if shared rule arbitration or upload contracts are affected.
- [ ] Run mirror verification and record exact results.
- [ ] Update `D:/ind-session/CURRENT_CODEX_STATE.yaml`, `WORKSPACE_STATUS_20260725.md` successor status, and dialogue log only through the documented state-maintenance process.

## Later Content-Audit Plan

## 3.2.R Contract Slice (2026-09-09)

- [x] Confirm Table 4's six extension titles from `node-extension-property_CN.xml`.
- [x] Add `core/ectd_32r_semantics.py` for conditional biologic/non-biologic 3.2.R semantics.
- [x] Add `schemas/ectd/ectd_32r_node_extension_contract.schema.json` for machine-readable contract shape.
- [x] Preserve Figure 2's XML skeleton at `data/regulations/normalized/cn_ectd_technical_specification.32r_figure2_skeleton.xml`.
- [x] Expose contract/schema/source references in existing 3.2.R rule details.
- [x] Sync the slice to the mirror and run mirror-focused verification.
- [ ] Add selected-path rule-detail drill-down for 3.2.R findings.
- [ ] Later validate per-extension content adequacy and document semantics; title/path correctness alone must not auto-pass content review.

After Phase 1 is stable, create a separate design and plan for:

- File parser dispatch by extension and content type.
- XML schema/DTD and XSL/STF validation against versioned official resources.
- PDF/DOCX/PPTX integrity and placeholder detection.
- Checksum verification against index and `index-md5.txt`.
- Cross-file title, language, lifecycle, study, product, and applicant consistency.
- Evidence graph connecting findings to file paths, XML nodes, page/region evidence, and source hashes.
- Risk levels and human-review queue with explicit `blocked_by_missing_evidence` handling.
- Incremental re-audit when one file changes, with invalidation based on dependency edges rather than rerunning every file.

## Verification and Governance

- Run `powershell -ExecutionPolicy Bypass -File D:/ind-session/autoind_worktree_preflight.ps1` before source edits and read `D:/ind-session/worktree_preflight_latest.md`.
- Keep production IND/eCTD logic separate from benchmark-neutral projection.
- Use the preferred interpreter: `D:/AutoIND-Pro/ind-compliance-ai/.venv/Scripts/python.exe`.
- Do not use PDF filename, page, sample id, exact text, or coordinates as production conditions.
- For parser/API/Markdown/frontend/test changes, sync relevant files to `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai` and run practical mirror verification.
- Do not claim package-level compliance for a partial upload.
- Before claiming completion, run fresh focused tests, coupled rule tests, and any required frontend/build checks; report exact outcomes and omitted broad suites with rationale.
- No release or commit boundary should include unrelated dirty-worktree changes.

## 2026-09-09 Execution Update

The application-root identifier slice is complete. The existing rule/schema coverage was confirmed in `cn_ectd_technical_specification.requirement_matrix.json` (`req_application_number_format`, clause 2.1.1) and `rules/rule_metadata.yaml` (`HR-ECTD-002`). The executable adapter now parses the application root as one type letter (`x`/`y`/`l`), four year digits, and five serial digits, and emits separate findings for total length, prefix, year, and serial errors. Historical years are accepted as lifecycle identifiers; a syntactically valid future year is a non-blocking manual-review warning rather than a hard failure.

The package inventory exposes the parsed application category/year/serial and the project overview renders them. The active `x202112345` fixture is valid and resolves to new-drug application, year 2021, serial `12345`. Primary and mirror naming/inventory/intake/support/upload tests pass `18 OK`; frontend lint/build and Python compilation pass in both workspaces.

Remaining Phase 1 work is to feed envelope `application-number`/`application-type` and module-1 evidence into a semantic second pass, then expand registration-activity, sequence, required-entry, and controlled-file rules. Do not infer current-year validity or application semantics from the folder name alone.

## 2026-09-09 Semantic Identity Execution Update

The application-type mapping now has an explicit evidence contract in `core/ectd_application_identity.py`. Strong evidence consists of envelope `application-type` and explicit application-form fields; module-1 titles/keywords are retained as weak evidence only. The adapter emits `supported`, `conflict`, or `insufficient_evidence` with evidence details and review flags, and `_build_workbench` exposes the result under `application_identity` for project-level presentation. This is deliberately separate from the generic regulatory rule JSON Schema, which validates catalog shape rather than executing eCTD semantics.

Primary semantic identity tests pass `6 OK`; existing focused eCTD package tests remain `18 OK`. Next: populate the adapter from parsed XML/form/title evidence and connect conflicts to the project finding detail panel.

## 2026-09-09 Attachment 1-2 Execution Update

The controlled vocabulary source files are now represented by an explicit semantic rule contract. The contract binds `application-type`, `product-type`, `regulatory-activity-type`, and `sequence-type` to their CV files and source clauses, records parsed XML evidence locations, preserves the `x/y/l` application-prefix mapping, and carries all rows from `depend-apt-rat-sqt.xml` for triplet compatibility checks. `core/ectd_controlled_vocabulary_rules.py` evaluates uploaded envelope values; `schemas/ectd/ectd_semantic_rule_contract.schema.json` defines the machine-readable contract.

The workbench application identity projection includes these per-sequence checks. Existing material assessment remains the authoritative finding evaluator; this projection adds traceability and a stable contract for later field-level UI review. Primary and mirror controlled-vocabulary/application-identity tests pass `10 OK`; focused eCTD package tests pass `24 OK`; frontend builds pass in both workspaces.

## 2026-09-09 Clinical-Trial Table 1 Sequence Semantics

The general sequence rules already covered four-digit numbering, `0000` start, contiguous history, related-sequence format/order, sequence-type vocabulary, description presence, and application/activity/sequence compatibility. The missing piece was an explicit machine-readable interpretation of the clinical-trial examples in technical-specification Table 1.

Added `core/ectd_sequence_semantics.py` and `schemas/ectd/ectd_sequence_semantic_contract.schema.json`. The contract records the ten observed rows (`0000`-`0009`) with related sequence, regulatory activity type, sequence type, and semantic description intent. The validator emits deterministic findings for sequence history and row-level semantic mismatches, while leaving absent envelope values to existing prerequisite/manual-review handling. `api/main.py` exposes the contract, per-application validation, and project findings; `ui/frontend/src/types.ts` declares the response fields. `rules/rule_metadata.yaml` registers `HR-ECTD-118`, and the normalized technical-specification requirement matrix adds `req_clinical_trial_sequence_table1_semantics` under clause 2.2.2.

Important interpretation: Table 1's `0000`-`0009` set is illustrative, not a hard upper bound. Valid lifecycle submissions after `0009` remain allowed when numbering continues by exactly one and the envelope semantics are valid. Sequence descriptions are evaluated by intent categories (initial submission, response, supplement, new indication/combination, safety update, potential serious safety risk), not brittle exact-literal matching.

Verification: primary focused eCTD/API suite `35 OK`; sequence and identity tests `13 OK`; JSON/schema shape and Python compilation pass; primary and mirror frontend production builds pass; mirror sequence/identity tests `13 OK`. The mirror legacy `.doc` ingestion test remains environment-blocked because no Word/LibreOffice converter is installed. The active `x202112345` sample contains only sequence `0000`, so it can exercise the initial-row path when envelope metadata is present but cannot demonstrate response/supplement rows without additional sequence fixtures.

## 2026-09-09 - Table 2, Table 3 and Sequence Quality Rules

The sequence semantic contract now distinguishes application types. Table 1 remains scoped to clinical-trial applications (`cnapt1`) with rows `0000`-`0009`; Table 2 is scoped to new-drug applications (`cnapt2`) with rows `0000`-`0008`, including the two initial responses to `0000`, manufacturing-process and analytical-method supplements, the new-indication submission, renewal, and their response relationships. The example ranges are non-exhaustive; sequences beyond the displayed endpoint produce manual-review guidance rather than automatic rejection.

Table 3 is modeled as a relationship-policy layer sourced from clause 2.4. Its examples are explanatory only. The current `depend-apt-rat-sqt.xml` matrix remains authoritative for legal application-type/regulatory-activity-type/sequence-type combinations. Matrix-compatible combinations not printed in Table 3 are allowed; incompatible observed triplets produce a deterministic finding.

The contract also exposes sequence-quality policy: description maximum 120 characters, required contact fields `name`/`phone`/`email`, and prohibited-use terms that suggest a sequence description is being used in place of a regulatory response, explanatory letter, or question to the authority. Length/contact failures are deterministic; prohibited-use and ambiguous intent are manual-review findings. `cv-sequence-type.xml` remains the source for `cnsqt1` Original, `cnsqt2` Response, `cnsqt3` Withdrawal, and `cnsqt4` Reformat.

Verification: primary focused eCTD/API suite `40 OK`; sequence/identity/ingestion subset `18 tests` with one pre-existing environment skip; Python compilation and contract JSON checks pass. Mirror sequence/identity tests pass `18 OK` under system Python; the mirror legacy `.doc` ingestion test is blocked by the missing Word/LibreOffice converter. The active `x202112345` sample remains a single-sequence scaffold and does not exercise the full Table 2 or Table 3 lifecycle.

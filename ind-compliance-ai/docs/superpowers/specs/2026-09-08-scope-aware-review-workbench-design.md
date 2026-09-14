# Scope-Aware Review Workbench Design

## Goal

Present single-document reviews and eCTD package reviews with different workbench surfaces while preserving existing PDF parsing, structural highlights, Markdown evidence, and rule findings.

## Product Decision

The project tree is navigation context, not a review result. It must have one owner. During upload it is shown as a temporary preview; after processing it moves to the left side of the review workbench as the submission navigator. The upload panel becomes a compact task summary and does not render the tree again.

The workbench has two presentation modes:

- `single_file`: retain the existing PDF viewer, structural boxes, Markdown view, and document-level conclusion panel.
- `project`: show the submission navigator on the left, a package review overview in the center, and scope-aware findings on the right. A PDF or other file viewer will be added incrementally as the selected-file contract becomes available; the project overview is the safe default when no file is selected.

Package mode is detected from `package_inventory` or an explicit sequence/application review scope. It must not depend only on the number of parsed files, because a valid package can contain one reviewable document plus XML and support files.

## Surface Responsibilities

### Upload task summary

Keep intake mode, processing status, progress, file count, and a compact result summary. Hide the full tree once processing finishes. The existing upload modes and single-file parsing behavior remain unchanged.

### Submission navigator

Render one project tree in the left workbench pane. It contains application, sequence, module, section, directory, and file nodes. File and directory nodes are selectable, and selection is exposed to the workbench state for the next selected-file viewer increment. Tree counts and source kind remain visible.

### Project overview

The center pane for project mode displays package scope, file/directory counts, finding counts, parse warnings, and a short explanation of the current audit boundary. It must not display the first PDF as though it represented the whole package, and it must not render a concatenated all-file Markdown document as the default review surface.

### Single-file content view

The current PDF viewer remains the center of the single-file review experience. Its page selector, structural bounding boxes, TOC navigation, and Markdown panel remain available.

### Findings panel

The right panel remains present in both modes. In project mode it presents package/sequence findings and the selected scope. In single-file mode it presents the existing document-level summary and rule checks. Existing rule navigation behavior is preserved.

## Data Contract

Reuse `WorkbenchPayload.review_scope`, `package_inventory`, `package_findings`, and `documents`. Extend the tree component with optional selection callbacks; do not add a second inventory endpoint. `review_scope` and `package_inventory` are authoritative for presentation mode.

## Verification

- Node contract tests prove package inventory forces project mode and a plain single PDF remains single-file mode.
- Frontend TypeScript/Vite production build passes in primary and mirror workspaces.
- Existing eCTD upload/inventory/rule tests remain green.
- A static inspection confirms `ProjectPackageTree` is rendered only once in the completed review workbench and once as a temporary upload preview before completion.

## Project Finding Presentation Follow-up

The project-mode right pane uses two finding layers: `structure_naming` and `file_content`. The first consumes package inventory structure/naming findings; the second consumes the existing rule-check items for files that entered the parser/rule pipeline. Cross-file consistency is deliberately kept out of the workbench right pane and remains owned by the page-level `ConsistencyBoard` at the bottom, where the reserved cross-module/cross-material/cross-sequence comparison surface already lives.

The default view is exception-first. Failed and manual-review items remain visible with rule/path drill-down actions. Passed items are counted in the layer header and placed behind a collapsed disclosure, while not-applicable and unsupported content are not presented as successful compliance. The three workbench columns share a responsive viewport height and independent scroll surfaces so a long finding list does not stretch one column or leave another visibly short.

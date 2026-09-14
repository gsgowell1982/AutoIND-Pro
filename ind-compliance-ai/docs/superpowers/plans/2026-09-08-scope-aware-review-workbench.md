# Scope-Aware Review Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Separate single-file and whole-project review presentation, remove duplicate project trees, and make the project workbench use a single left navigation tree with project-level overview and findings.

**Architecture:** Use `review_scope` and `package_inventory` as the presentation mode contract. Keep the upload tree only during intake; after completion render the tree once inside the project workbench left pane. Keep the existing single-file PDF/Markdown/review layout and replace only the project-mode center surface with a package overview.

**Tech Stack:** React 19, Ant Design, TypeScript, Vite, existing JavaScript scope helper, Python unittest regression suite.

---

### Task 1: Add review-scope contract tests

**Files:**
- Create: `ui/frontend/src/workbenchReviewScope.test.mjs`
- Modify: `ui/frontend/src/workbenchReviewScope.js`

- [x] Write Node tests for a package inventory with one PDF and support files, asserting `mode === 'project'` and `isProjectReview === true`.
- [x] Write a second test for one PDF without package inventory, asserting `mode === 'single_file'` and `isSingleFile === true`.
- [x] Run `node --test ui/frontend/src/workbenchReviewScope.test.mjs` and confirm the package test fails against the current file-count-only implementation.
- [x] Update `resolveWorkbenchReviewScope` so package inventory or explicit `sequence`/`application` scope forces project mode, while a plain single PDF remains single-file mode.
- [x] Run the same Node test and confirm both cases pass.

### Task 2: Make the project tree selectable and reusable as a navigator

**Files:**
- Modify: `ui/frontend/src/components/ProjectPackageTree.tsx`

- [x] Add optional `selectedPath` and `onPathSelect(path, kind)` props.
- [x] Keep the existing directory/file counts, source-kind tag, virtual scrolling, and default expansion.
- [x] Set `selectable` and `selectedKeys` only when a selection callback is provided; map `file:` and `dir:` keys back to normalized relative paths.
- [x] Run the frontend TypeScript build to catch Ant Design tree type errors.

### Task 3: Remove the completed-upload duplicate tree

**Files:**
- Modify: `ui/frontend/src/components/UploadPreprocessPanel.tsx`

- [x] Render the temporary project tree only while the job is absent, queued, or processing.
- [x] After completion, retain the status, progress result, and file list but omit the full tree so the workbench owns the completed-project navigation tree.
- [x] Keep the selected-folder preview tree before submission.
- [x] Run the frontend build and verify the upload panel still renders the single-file, ZIP, and folder intake modes.

### Task 4: Add project-mode overview and move the tree into the left workbench pane

**Files:**
- Modify: `ui/frontend/src/components/AuditWorkbench.tsx`

- [x] Derive `isProjectReview` from `getReviewPanelVisibility(workbench).scope`.
- [x] Remove the standalone tree above the three-column workbench.
- [x] In project mode, render one `ProjectPackageTree` inside the left pane with selected-path state and a compact package summary.
- [x] In single-file mode, preserve the existing PDF viewer and structural highlight behavior in the left pane.
- [x] In project mode, replace the center Markdown panel with a project overview showing scope, file/directory totals, package finding totals, and an explanation that a file must be selected for file-specific evidence.
- [x] Preserve the existing Markdown panel in single-file mode.
- [x] Keep the right findings and rule navigation panel available in both modes, using existing `package_findings` and `rule_checks` data.
- [x] Run the frontend build and inspect the rendered structure statically to confirm no duplicate completed-project tree remains.

### Task 5: Synchronize and verify

**Files:**
- Synchronize the task-owned frontend, test, design, and plan files to `D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai`.
- Update: `D:\ind-session\WORKSPACE_STATUS_20260725.md`
- Update: `D:\ind-session\ACTIVE_WORK_CONTEXT.md`

- [x] Run `node --test ui/frontend/src/workbenchReviewScope.test.mjs` in primary and mirror.
- [x] Run the focused eCTD unittest suite in primary and mirror.
- [x] Run `npm run build` in primary and mirror frontend directories.
- [x] Run `git diff --check` on task-owned files in both workspaces.
- [x] Record the scope-aware workbench checkpoint and known limitation that selected-file content switching is the next increment.

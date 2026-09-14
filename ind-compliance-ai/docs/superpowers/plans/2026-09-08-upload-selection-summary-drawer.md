# Upload Selection Summary and File Detail Drawer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the overflowing inline project-file list with a compact selection summary and searchable detail drawer while preserving the single-document uploader.

**Architecture:** A pure JavaScript helper normalizes selected file records and produces summary/filter data. `UploadPreprocessPanel` uses that helper, hides the Ant Design inline list only for project modes, and renders the details in a bounded `Drawer`. CSS gives the drawer list its own scroll surface; existing project-tree and API contracts remain unchanged.

**Tech Stack:** React 19, TypeScript, Ant Design 5, Vite, Node's built-in test runner.

---

### Task 1: Define selection summary behavior with failing tests

**Files:**
- Create: `ui/frontend/src/uploadSelectionSummary.test.mjs`
- Create: `ui/frontend/src/uploadSelectionSummary.js`

- [x] **Step 1: Write the failing tests**

Add tests for one project selection containing duplicate directory paths and a case-insensitive search query. Assert that the summary returns unique directory count, file count, root label, total bytes, and that filtering matches relative paths.

- [x] **Step 2: Run the tests and verify the expected failure**

Run `node --test ui/frontend/src/uploadSelectionSummary.test.mjs` from `D:\AutoIND-Pro\ind-compliance-ai\ui\frontend`.

Expected: the test fails because `uploadSelectionSummary.js` does not export the requested helper functions.

- [x] **Step 3: Implement the minimal pure helpers**

Implement `buildUploadSelectionSummary({ mode, files, directoryPaths })` and `filterUploadSelectionFiles(files, query)`. Normalize backslashes to slashes, remove empty path segments, de-duplicate directory paths, calculate total bytes, and derive a single root label only when all paths share one first segment.

- [x] **Step 4: Run the tests and verify they pass**

Run the same Node command. Expected: all summary/filter tests pass.

### Task 2: Integrate project summary and detail drawer

**Files:**
- Create: `ui/frontend/src/uploadSelectionSummary.d.ts`
- Modify: `ui/frontend/src/components/UploadPreprocessPanel.tsx`

- [x] **Step 1: Add the type contract**

Declare the helper input/output types so strict TypeScript can import the JavaScript module without weakening compiler settings.

- [x] **Step 2: Add drawer state and derived records**

Import `Drawer`, `Empty`, `Input`, `SearchOutlined`, and `UnorderedListOutlined`. Add `selectionDrawerOpen` and `selectionQuery` state. Convert selected browser files to `{ path, name, size }` records, derive the summary, and filter records using the helper.

- [x] **Step 3: Replace the overflowing inline project list**

Set `showUploadList={intakeMode === 'single'}` on `Upload.Dragger`. For `zip` and `directory`, render a compact summary below the dragger with the count tags and a link-style `View file list` button. Keep the existing single-file list behavior untouched.

- [x] **Step 4: Render the bounded read-only drawer**

Render an Ant Design `Drawer` outside the two-column row. Include a search input, summary tags, and a scrollable `List` of normalized relative paths with byte sizes. Show `Empty` when no records match. Closing the drawer clears the query.

- [x] **Step 5: Run the frontend type/build check**

Run `npm run build` in `D:\AutoIND-Pro\ind-compliance-ai\ui\frontend`. Expected: TypeScript and Vite complete successfully.

### Task 3: Style, mirror sync, and regression verification

**Files:**
- Modify: `ui/frontend/src/index.css`
- Sync: the same frontend files into `D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai`

- [x] **Step 1: Add bounded detail-list styles**

Add classes for the summary row and drawer file list, with `max-height`, `overflow: auto`, `min-width: 0`, and long-path ellipsis. Do not change the established equal-height upload-column token.

- [x] **Step 2: Mirror the touched frontend files**

Copy `UploadPreprocessPanel.tsx`, `uploadSelectionSummary.js`, `uploadSelectionSummary.d.ts`, `uploadSelectionSummary.test.mjs`, and `index.css` to the mirror at the same relative paths.

- [x] **Step 3: Run primary and mirror verification**

Run the helper tests, the existing workbench Node tests, the focused eCTD Python suite, and `npm run build` in both frontend directories. Expected: helper tests pass, workbench tests pass `6 OK`, eCTD tests pass `12 OK`, and both builds succeed.

- [x] **Step 4: Check synchronization and whitespace**

Run `git diff --check` on the touched frontend files in both repositories and compare SHA-256 hashes for each synchronized file. Expected: no whitespace errors and matching hashes.

### Task 4: Record the checkpoint

**Files:**
- Modify: `D:\ind-session\CURRENT_CODEX_STATE.yaml`
- Modify: `D:\ind-session\WORKSPACE_STATUS_20260725.md`
- Modify: `D:\ind-session\ACTIVE_WORK_CONTEXT.md`

- [x] **Step 1: Record the product decision and verification**

Add a dated checkpoint stating that project-mode inline file rows are replaced by summary plus drawer, single-file mode is preserved, and both repositories passed the listed tests/builds.

- [x] **Step 2: Parse the YAML state file**

Load `CURRENT_CODEX_STATE.yaml` with the project virtualenv's PyYAML and assert that the new checkpoint status is `implemented_verified_mirror_synced`.

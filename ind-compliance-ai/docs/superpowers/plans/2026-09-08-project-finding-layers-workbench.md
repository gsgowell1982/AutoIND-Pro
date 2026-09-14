# Project Finding Layers Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the three workbench result panes and present project review findings as actionable exceptions across package structure/naming and single-file content, while keeping the reserved cross-file consistency board at page level.

**Architecture:** Keep the existing `WorkbenchPayload` contract. Add a small pure presentation adapter that classifies findings and returns counts plus actionable items; render the two project layers in project mode on the right. Keep `ConsistencyBoard` as the only page-level cross-file surface. Use CSS flex and a shared viewport height so all three columns have equal frames with independent scrolling.

**Tech Stack:** React 19, TypeScript, Ant Design, Vite, Node test runner, existing `WorkbenchPayload` and rule presentation helpers.

---

### Task 1: Add the finding-layer presentation contract

**Files:**
- Create: `ui/frontend/src/projectFindingLayers.js`
- Create: `ui/frontend/src/projectFindingLayers.test.mjs`

- [x] **Step 1: Write failing tests** for structure/naming findings, content rule findings, manual-review statuses, hidden passes, and the separation from the page-level consistency board.

```js
import test from 'node:test'
import assert from 'node:assert/strict'
import { buildProjectFindingLayers } from './projectFindingLayers.js'

test('keeps package failures and manual review visible while collapsing passes', () => {
  const result = buildProjectFindingLayers({
    package_findings: [
      { rule_id: 'HR-ECTD-001', status: 'fail', severity: 'error', relative_path: 'x/0000/m3', message: 'missing module' },
      { rule_id: 'HR-ECTD-002', status: 'pass', relative_path: 'x', message: 'valid root' },
      { rule_id: 'HR-ECTD-003', status: 'review_required', severity: 'warning', relative_path: 'x/0000/index.xml', message: 'manual check' },
    ],
    rule_checks: { items: [] },
  })
  const layer = result.layers.find((item) => item.id === 'structure_naming')
  assert.deepEqual(layer.counts, { fail: 1, review: 1, pass: 1 })
  assert.equal(layer.actionable.length, 2)
  assert.equal(layer.passed.length, 1)
})

test('classifies content rules separately and reserves cross-file consistency', () => {
  const result = buildProjectFindingLayers({
    package_findings: [],
    rule_checks: {
      items: [
        { rule_id: 'SR-001', category: 'content', status: 'warn', message: 'content warning' },
        { rule_id: 'SR-002', category: 'format', status: 'fail', message: 'format failure' },
        { rule_id: 'SR-003', category: 'document', status: 'pass', message: 'content pass' },
      ],
    },
  })
  const content = result.layers.find((item) => item.id === 'file_content')
  assert.deepEqual(content.counts, { fail: 1, review: 1, pass: 1 })
  assert.deepEqual(content.actionable.map((item) => item.rule_id), ['SR-001', 'SR-002'])
  const consistency = result.crossFileConsistency
  assert.equal(consistency.status, 'reserved')
  assert.equal(consistency.counts.actionable, 0)
})
```

- [x] **Step 2: Run the tests and verify RED.**

Run: `node --test ui/frontend/src/projectFindingLayers.test.mjs`

Expected: FAIL because `projectFindingLayers.js` does not exist.

- [x] **Step 3: Implement the minimal adapter.** Normalize `fail`, `warn`, `review_required`, `manual_review`, and `pass`; map warnings/manual-review to `review`; preserve original finding fields; count unknown statuses as actionable review items only when they are not `pass`/`na`. Return the cross-file interface separately so it cannot be rendered in the right-pane layer list.

- [x] **Step 4: Run the tests and verify GREEN.**

Run: `node --test ui/frontend/src/projectFindingLayers.test.mjs`

Expected: `2` tests passed, `0` failed.

### Task 2: Build the layered project conclusion panel

**Files:**
- Modify: `ui/frontend/src/components/AuditWorkbench.tsx`
- Modify: `ui/frontend/src/types.ts` only if TypeScript needs the adapter result type

- [x] **Step 1: Write the component against the adapter.** Add `ProjectFindingLayersPanel` with an actionable-first view, three layer cards, compact count tags, and a collapsible passed-items section. Use the existing `focusRule` callback for rule findings and the existing package-tree selection setter for relative paths.

- [x] **Step 2: Verify the component contract fails before wiring.** Run the frontend TypeScript build after importing the not-yet-created component symbol; confirm the expected missing-export/type failure.

- [x] **Step 3: Implement the panel.** In project mode show:
  - structure/naming actionable items from `package_findings`;
  - file content actionable items from `rule_checks.items` with rule detail navigation;
  - no cross-file consistency card; the page-level `ConsistencyBoard` remains the owner of that surface.
  Successful items render only inside `Collapse` and never in the default actionable list.

- [x] **Step 4: Run the frontend build and inspect the project-mode branch.** Confirm the existing single-file right panel remains available and the project panel does not claim content validation for unparsed support files.

### Task 3: Equalize the three workbench result frames

**Files:**
- Modify: `ui/frontend/src/index.css`
- Modify: `ui/frontend/src/components/AuditWorkbench.tsx`

- [x] **Step 1: Add the shared viewport contract.** Define `--workbench-panel-height: clamp(560px, calc(100vh - 300px), 760px)`; make `.workbench-col` a flex column with that height; give `.workbench-col > .ant-pro-card-body` `min-height: 0`, `display: flex`, `flex-direction: column`; add `.workbench-scroll-surface` with `flex: 1`, `min-height: 0`, and `overflow: auto`.

- [x] **Step 2: Wrap each column's result content in the shared scroll surface.** Keep toolbars/summary headers above it, set the left page selector `width: 100%`, and let middle/right controls use available width without hard-coded short widths.

- [x] **Step 3: Run the production build and `git diff --check`.** Confirm no layout CSS/type errors.

### Task 4: Synchronize and verify

**Files:**
- Synchronize task-owned frontend, test, plan, and design files to `D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai`.
- Update: `D:\ind-session\CURRENT_CODEX_STATE.yaml`
- Update: `D:\ind-session\WORKSPACE_STATUS_20260725.md`
- Update: `D:\ind-session\ACTIVE_WORK_CONTEXT.md`

- [x] **Step 1: Run primary tests.** `node --test ui/frontend/src/projectFindingLayers.test.mjs`; focused eCTD unittest suite; `npm run build` under `ui/frontend`.
- [x] **Step 2: Run mirror tests.** Repeat the same commands in the mirror workspace.
- [x] **Step 3: Run static checks.** `git diff --check` for task-owned files and SHA-256 compare primary/mirror files.
- [x] **Step 4: Record the checkpoint.** Document the two right-pane finding layers, page-level consistency-board ownership, default exception-only view, equal-height panes, and the reserved consistency interface. Preserve the next boundary: selected-file content evidence and cross-file logic implementation.

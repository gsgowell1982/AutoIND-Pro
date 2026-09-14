import test from 'node:test'
import assert from 'node:assert/strict'

import { buildProjectFindingLayers } from './projectFindingLayers.js'

test('keeps package failures and manual review visible while collapsing passes', () => {
  const result = buildProjectFindingLayers({
    package_findings: [
      {
        rule_id: 'HR-ECTD-001',
        status: 'fail',
        severity: 'error',
        relative_path: 'x/0000/m3',
        message: 'missing module',
      },
      { rule_id: 'HR-ECTD-002', status: 'pass', relative_path: 'x', message: 'valid root' },
      {
        rule_id: 'HR-ECTD-003',
        status: 'review_required',
        severity: 'warning',
        relative_path: 'x/0000/index.xml',
        message: 'manual check',
      },
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

test('treats an executed empty layer as clear instead of unavailable', () => {
  const result = buildProjectFindingLayers({
    package_findings: [],
    rule_checks: { items: [] },
  })
  assert.equal(result.layers.find((item) => item.id === 'structure_naming').status, 'clear')
  assert.equal(result.layers.find((item) => item.id === 'file_content').status, 'clear')
})

test('keeps cross-file consistency out of the workbench right-pane layers', () => {
  const result = buildProjectFindingLayers({
    package_findings: [],
    rule_checks: { items: [] },
  })
  assert.deepEqual(result.layers.map((item) => item.id), ['structure_naming', 'file_content'])
  assert.equal(result.crossFileConsistency.status, 'reserved')
})

test('focuses findings on a selected file while retaining project-level findings', () => {
  const result = buildProjectFindingLayers(
    {
      package_findings: [
        { rule_id: 'P-1', status: 'fail', relative_path: 'x/0000/m3/a.pdf', message: 'selected file' },
        { rule_id: 'P-2', status: 'fail', relative_path: 'x/0000/m3', message: 'selected directory' },
        { rule_id: 'P-3', status: 'fail', relative_path: 'x/0000/m4/a.pdf', message: 'other file' },
        { rule_id: 'P-4', status: 'fail', message: 'project level' },
      ],
      rule_checks: { items: [] },
    },
    { selectedPath: 'x/0000/m3/a.pdf' },
  )

  const ids = result.layers[0].actionable.map((item) => item.rule_id)
  assert.deepEqual(ids, ['P-1', 'P-2', 'P-4'])
  assert.equal(result.focusedPath, 'x/0000/m3/a.pdf')
})

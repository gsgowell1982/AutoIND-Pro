import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildSingleFileReviewSummary,
  getReviewPanelVisibility,
  resolveWorkbenchReviewScope,
} from '../src/workbenchReviewScope.js'

function buildWorkbench(overrides = {}) {
  return {
    pdf_document: {
      file_id: 'file-ectd-guide',
      filename: 'eCTD实施指南.pdf',
      pages: [{ page_number: 1 }],
    },
    demo_sample: null,
    demo_summary: { summary: {} },
    demo_run: { summary: {} },
    rule_checks: {
      enabled: true,
      items: [],
      structure_audit_records: [],
      summary: {},
      risk_count: 0,
      message: '当前规则结果由材料审阅契约与确定性规则引擎生成。',
    },
    ...overrides,
  }
}

test('treats a normal one-PDF workbench as single-file review even when demo projections exist', () => {
  const scope = resolveWorkbenchReviewScope(buildWorkbench())

  assert.equal(scope.mode, 'single_file')
  assert.equal(scope.filename, 'eCTD实施指南.pdf')
  assert.equal(scope.isControlledDemo, false)
  assert.equal(scope.isProjectReview, false)
})

test('keeps controlled demo mode when synthetic demo metadata is present', () => {
  const scope = resolveWorkbenchReviewScope(
    buildWorkbench({
      pdf_document: null,
      demo_sample: {
        sample_id: 'phase-a-demo',
        synthetic: true,
        sample_kind: 'two_sequence_ectd_batch_with_identity_conflict',
      },
    }),
  )

  assert.equal(scope.mode, 'controlled_demo')
  assert.equal(scope.isControlledDemo, true)
  assert.equal(scope.isProjectReview, false)
})

test('single-file review hides demo-only panels while preserving file result modules', () => {
  const visibility = getReviewPanelVisibility(
    buildWorkbench({
      rule_checks: {
        enabled: true,
        items: [{ status: 'warn' }],
        structure_audit_records: [{ alignment_ready: false, filename: 'eCTD实施指南.pdf' }],
        summary: {},
        risk_count: 0,
        message: '当前规则结果由材料审阅契约与确定性规则引擎生成。',
      },
    }),
  )

  assert.equal(visibility.showDemoPanels, false)
  assert.equal(visibility.showSingleFileSummary, true)
  assert.equal(visibility.showProjectReadinessPanels, false)
  assert.equal(visibility.showStructureAudit, true)
  assert.equal(visibility.showRuleChecks, true)
})

test('controlled demo visibility still shows demo-only panels', () => {
  const visibility = getReviewPanelVisibility(
    buildWorkbench({
      pdf_document: null,
      demo_sample: {
        sample_id: 'phase-a-demo',
        synthetic: true,
        sample_kind: 'two_sequence_ectd_batch_with_identity_conflict',
      },
    }),
  )

  assert.equal(visibility.showDemoPanels, true)
  assert.equal(visibility.showSingleFileSummary, false)
  assert.equal(visibility.showProjectReadinessPanels, true)
})

test('single-file summary leads with file-scoped format and rule modules only', () => {
  const summary = buildSingleFileReviewSummary(
    buildWorkbench({
      rule_checks: {
        enabled: true,
        items: [{ status: 'warn' }, { status: 'pass' }, { status: 'na' }],
        structure_audit_records: [
          { alignment_ready: false, filename: 'eCTD实施指南.pdf' },
          { alignment_ready: true, filename: 'eCTD实施指南.pdf' },
        ],
        summary: { warn_rules: 1, pass_rules: 1, na_rules: 1 },
        risk_count: 0,
        message: '当前规则结果由材料审阅契约与确定性规则引擎生成。',
      },
      content_consistency: {
        summary: {
          check_count: 1,
          issue_count: 1,
          prerequisite_required_count: 0,
          deterministic_rule_verdict_count: 0,
        },
      },
    }),
  )

  assert.equal(summary.title, '单个 PDF 文件审阅')
  assert.equal(summary.filename, 'eCTD实施指南.pdf')
  assert.match(summary.conclusion, /本次审阅对象为单个 PDF 文件/)
  assert.deepEqual(
    summary.modules.map((module) => module.name),
    ['格式与结构', '规则与前置条件'],
  )
  assert.equal(summary.modules[0].status, 'review_required')
  assert.ok(summary.modules.every((module) => !module.name.includes('内容一致性')))
  assert.ok(summary.modules.every((module) => !module.summary.includes('一致性复核')))
})

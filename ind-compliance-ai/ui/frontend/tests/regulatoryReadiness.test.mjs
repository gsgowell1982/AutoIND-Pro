import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildRegulatoryReadinessMatrixRows,
  getRegulatoryReadinessCoverageLabel,
  getRegulatoryReadinessStatusColor,
  getRegulatoryReadinessTriageLabel,
} from '../src/regulatoryReadiness.js'

const SAMPLE_SOURCES = [
  {
    regulation_id: 'cn_ectd_validation_standard',
    source_file: 'eCTD validation standard.pdf',
    coverage_status: 'closed',
    clause_count: 149,
    covered_count: 146,
    partial_count: 0,
    deferred_count: 0,
    citation_only_count: 3,
    rule_candidate_count: 0,
    direct_ish_candidate_count: 0,
    requirement_count: 0,
    direct_rule_draft_count: 0,
    recommended_product_role: 'deterministic_validation_backbone',
    default_triage: 'deterministic',
    automation_boundary: 'Keep deterministic validation-standard rules closed.',
  },
  {
    regulation_id: 'cn_drug_registration_classification_and_dossier_requirements',
    source_file: 'dossier requirements.doc',
    coverage_status: 'parsed_not_coverage_closed',
    clause_count: 17,
    covered_count: 0,
    partial_count: 0,
    deferred_count: 0,
    citation_only_count: 0,
    rule_candidate_count: 17,
    direct_ish_candidate_count: 5,
    requirement_count: 9,
    direct_rule_draft_count: 3,
    recommended_product_role: 'dossier_checklist_and_applicability_backbone',
    default_triage: 'prerequisite_required',
    automation_boundary: 'Require application facts before hard applicability judgment.',
  },
]

test('returns customer-readable labels for readiness status and triage boundaries', () => {
  assert.equal(getRegulatoryReadinessCoverageLabel('closed'), '已闭环')
  assert.equal(getRegulatoryReadinessCoverageLabel('traceability_closed'), '追溯闭环')
  assert.equal(getRegulatoryReadinessCoverageLabel('parsed_not_coverage_closed'), '已解析，未覆盖闭环')
  assert.equal(getRegulatoryReadinessTriageLabel('deterministic'), '可确定性判定')
  assert.equal(getRegulatoryReadinessTriageLabel('prerequisite_required'), '需要前置条件')
  assert.equal(getRegulatoryReadinessTriageLabel('human_review'), '建议人工审核')
  assert.equal(getRegulatoryReadinessStatusColor('closed'), 'green')
  assert.equal(getRegulatoryReadinessStatusColor('parsed_not_coverage_closed'), 'gold')
})

test('builds source readiness matrix rows without inventing compliance verdicts', () => {
  const rows = buildRegulatoryReadinessMatrixRows(SAMPLE_SOURCES)

  assert.deepEqual(
    rows.map((row) => row.regulationId),
    [
      'cn_ectd_validation_standard',
      'cn_drug_registration_classification_and_dossier_requirements',
    ],
  )
  assert.deepEqual(rows[0], {
    key: 'cn_ectd_validation_standard',
    regulationId: 'cn_ectd_validation_standard',
    sourceFile: 'eCTD validation standard.pdf',
    coverageStatus: 'closed',
    coverageLabel: '已闭环',
    statusColor: 'green',
    clauseSummary: '条款 149 | 已覆盖 146 | 引用/统计 3',
    ruleSummary: '候选 0 | 直接草案 0',
    productRoleLabel: '确定性验证骨架',
    triage: 'deterministic',
    triageLabel: '可确定性判定',
    triageColor: 'green',
    automationBoundary: 'Keep deterministic validation-standard rules closed.',
  })
  assert.equal(rows[1].clauseSummary, '条款 17 | 需求 9')
  assert.equal(rows[1].ruleSummary, '候选 17 | 直接倾向 5 | 直接草案 3')
  assert.equal(rows[1].productRoleLabel, '资料清单与适用性骨架')
  assert.equal(rows[1].triageLabel, '需要前置条件')
})

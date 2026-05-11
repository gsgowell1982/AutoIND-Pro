import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildRuleDetailDisplayRows,
  getMatchStrengthLabel,
} from '../src/ruleDetailPresentation.js'

test('renders regulation severity once as a localized chip row', () => {
  const rows = buildRuleDetailDisplayRows({
    requirement_id: 'cn_ectd_validation_standard:req_pdf_link_action_whitelist',
    citation_anchor: 'cn_ectd_validation_standard#sec_3_1_4',
    source_clause_id: 'sec_3_1_4',
    regulation_severity: '错误',
    match_strength: 'pdf_disallowed_hyperlink_action_detected',
  })

  assert.equal(rows.filter((row) => row.label === '法规级别').length, 1)
  assert.deepEqual(rows.find((row) => row.label === '法规级别'), {
    kind: 'tag',
    label: '法规级别',
    value: '错误',
    color: 'red',
  })
})

test('hides backend requirement and clause trace rows from customer-facing details', () => {
  const rows = buildRuleDetailDisplayRows({
    requirement_id: 'cn_ectd_validation_standard:req_pdf_hyperlink_action_whitelist',
    citation_anchor: 'cn_ectd_validation_standard#sec_6_11',
    source_clause_id: 'cn_ectd_validation_standard#sec_6_11',
    regulation_severity: '错误',
    match_strength: 'pdf_disallowed_hyperlink_action_detected',
  })

  assert.equal(rows.some((row) => row.label === '规则要求'), false)
  assert.equal(rows.some((row) => row.label === '条款溯源'), false)
  assert.equal(
    rows.some((row) =>
      String(row.value).includes('cn_ectd_validation_standard:req_pdf_hyperlink_action_whitelist'),
    ),
    false,
  )
  assert.equal(rows.some((row) => String(row.value).includes('cn_ectd_validation_standard#sec_6_11')), false)
})

test('localizes known technical match-strength codes and hides raw internal ids', () => {
  assert.equal(getMatchStrengthLabel('pdf_disallowed_hyperlink_action_detected'), '检测到不允许的 PDF 超链接动作')
  assert.equal(getMatchStrengthLabel('unexpected_internal_code'), null)
})

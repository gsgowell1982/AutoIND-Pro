import test from 'node:test'
import assert from 'node:assert/strict'

import {
  DEFAULT_RULE_CHECK_STATUS_FILTER,
  filterRuleCheckItems,
  getRuleCheckFilterLabel,
  getRuleCheckStatusFilterOptions,
} from '../src/ruleCheckFilters.js'

const SAMPLE_ITEMS = [
  { rule_id: 'HR-001', status: 'fail' },
  { rule_id: 'SR-001', status: 'warn' },
  { rule_id: 'SR-002', status: 'pass' },
  { rule_id: 'SR-003', status: 'na' },
  { rule_id: 'SR-004', status: 'na' },
]

test('defaults to actionable rule checks and hides not applicable rows', () => {
  assert.equal(DEFAULT_RULE_CHECK_STATUS_FILTER, 'actionable')
  assert.deepEqual(
    filterRuleCheckItems(SAMPLE_ITEMS, DEFAULT_RULE_CHECK_STATUS_FILTER).map((item) => item.rule_id),
    ['HR-001', 'SR-001', 'SR-002'],
  )
})

test('filters rule checks by clicked summary category', () => {
  assert.deepEqual(filterRuleCheckItems(SAMPLE_ITEMS, 'fail').map((item) => item.rule_id), ['HR-001'])
  assert.deepEqual(filterRuleCheckItems(SAMPLE_ITEMS, 'warn').map((item) => item.rule_id), ['SR-001'])
  assert.deepEqual(filterRuleCheckItems(SAMPLE_ITEMS, 'pass').map((item) => item.rule_id), ['SR-002'])
  assert.deepEqual(filterRuleCheckItems(SAMPLE_ITEMS, 'na').map((item) => item.rule_id), ['SR-003', 'SR-004'])
})

test('risk filter follows backend risk semantics and shows fail plus warn rows', () => {
  assert.deepEqual(filterRuleCheckItems(SAMPLE_ITEMS, 'risk').map((item) => item.rule_id), ['HR-001', 'SR-001'])
})

test('unknown filter falls back to actionable rows', () => {
  assert.deepEqual(filterRuleCheckItems(SAMPLE_ITEMS, 'unknown').map((item) => item.rule_id), [
    'HR-001',
    'SR-001',
    'SR-002',
  ])
})

test('builds clickable summary filter options from rule summary counts', () => {
  assert.deepEqual(
    getRuleCheckStatusFilterOptions(
      {
        hard_failures: 2,
        soft_risks: 6,
        pass_rules: 29,
        na_rules: 161,
      },
      8,
    ),
    [
      { key: 'fail', label: '硬规则失败', count: 2, color: 'red' },
      { key: 'warn', label: '软规则风险', count: 6, color: 'orange' },
      { key: 'pass', label: '通过', count: 29, color: 'green' },
      { key: 'na', label: '不适用', count: 161, color: 'default' },
    ],
  )
})

test('filters rule checks within a clicked group status scope', () => {
  assert.deepEqual(
    filterRuleCheckItems(SAMPLE_ITEMS, 'na', ['HR-001', 'SR-003']).map((item) => item.rule_id),
    ['SR-003'],
  )
  assert.deepEqual(
    filterRuleCheckItems(SAMPLE_ITEMS, 'risk', ['HR-001', 'SR-001', 'SR-002']).map((item) => item.rule_id),
    ['HR-001', 'SR-001'],
  )
})

test('returns customer-readable labels for active rule check filters', () => {
  assert.equal(getRuleCheckFilterLabel('actionable'), '需处理与通过规则')
  assert.equal(getRuleCheckFilterLabel('na'), '不适用规则')
  assert.equal(getRuleCheckFilterLabel('risk'), '风险项')
})

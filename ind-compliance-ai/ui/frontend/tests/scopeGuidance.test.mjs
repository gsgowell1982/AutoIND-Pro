import test from 'node:test'
import assert from 'node:assert/strict'

import { resolveGuidanceRuleGroupFocusTarget } from '../src/scopeGuidance.js'

test('focuses the first non-na activity rule when guidance points to activity scope', () => {
  const target = resolveGuidanceRuleGroupFocusTarget({
    guidanceRuleGroups: ['activity'],
    ruleItems: [
      { rule_id: 'HR-ECTD-015', status: 'pass', scope: 'sequence' },
      { rule_id: 'SR-ECTD-005', status: 'warn', scope: 'activity' },
      { rule_id: 'SR-ECTD-006', status: 'pass', scope: 'application' },
    ],
  })

  assert.equal(target, 'SR-ECTD-005')
})

test('prefers a failing application rule over passing rules when multiple rules share the same scope', () => {
  const target = resolveGuidanceRuleGroupFocusTarget({
    guidanceRuleGroups: ['application'],
    ruleItems: [
      { rule_id: 'SR-ECTD-006', status: 'pass', scope: 'application' },
      { rule_id: 'HR-APP-001', status: 'fail', scope: 'application' },
    ],
  })

  assert.equal(target, 'HR-APP-001')
})

test('falls back to the next requested scope when the first requested scope has no applicable rule', () => {
  const target = resolveGuidanceRuleGroupFocusTarget({
    guidanceRuleGroups: ['activity', 'application'],
    ruleItems: [
      { rule_id: 'SR-ECTD-005', status: 'na', scope: 'activity' },
      { rule_id: 'SR-ECTD-006', status: 'warn', scope: 'application' },
    ],
  })

  assert.equal(target, 'SR-ECTD-006')
})

test('returns null when no requested guidance scope has an applicable rule', () => {
  const target = resolveGuidanceRuleGroupFocusTarget({
    guidanceRuleGroups: ['activity', 'application'],
    ruleItems: [
      { rule_id: 'SR-ECTD-005', status: 'na', scope: 'activity' },
      { rule_id: 'SR-ECTD-006', status: 'na', scope: 'application' },
    ],
  })

  assert.equal(target, null)
})

export const DEFAULT_RULE_CHECK_STATUS_FILTER = 'actionable'

const RULE_CHECK_FILTER_LABELS = {
  actionable: '需处理与通过规则',
  fail: '硬规则失败',
  warn: '软规则风险',
  pass: '通过规则',
  na: '不适用规则',
  risk: '风险项',
}

export function filterRuleCheckItems(items, filterKey = DEFAULT_RULE_CHECK_STATUS_FILTER, scopedRuleIds = null) {
  const sourceItems = Array.isArray(items) ? items : []
  const scopedRuleIdSet = Array.isArray(scopedRuleIds) ? new Set(scopedRuleIds) : null
  const scopedItems = scopedRuleIdSet
    ? sourceItems.filter((item) => scopedRuleIdSet.has(item?.rule_id))
    : sourceItems

  switch (filterKey) {
    case 'fail':
      return scopedItems.filter((item) => item?.status === 'fail')
    case 'warn':
      return scopedItems.filter((item) => item?.status === 'warn')
    case 'pass':
      return scopedItems.filter((item) => item?.status === 'pass')
    case 'na':
      return scopedItems.filter((item) => item?.status === 'na')
    case 'risk':
      return scopedItems.filter((item) => item?.status === 'fail' || item?.status === 'warn')
    case 'actionable':
    default:
      return scopedItems.filter((item) => item?.status !== 'na')
  }
}

export function getRuleCheckStatusFilterOptions(summary = {}) {
  return [
    { key: 'fail', label: '硬规则失败', count: summary?.hard_failures ?? 0, color: 'red' },
    { key: 'warn', label: '软规则风险', count: summary?.soft_risks ?? 0, color: 'orange' },
    { key: 'pass', label: '通过', count: summary?.pass_rules ?? 0, color: 'green' },
    { key: 'na', label: '不适用', count: summary?.na_rules ?? 0, color: 'default' },
  ]
}

export function getRuleCheckFilterLabel(filterKey = DEFAULT_RULE_CHECK_STATUS_FILTER) {
  return RULE_CHECK_FILTER_LABELS[filterKey] ?? RULE_CHECK_FILTER_LABELS[DEFAULT_RULE_CHECK_STATUS_FILTER]
}

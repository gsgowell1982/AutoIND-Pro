function getRuleStatusPriority(status) {
  switch (status) {
    case 'fail':
      return 0
    case 'warn':
      return 1
    case 'pass':
      return 2
    default:
      return 3
  }
}

export function resolveGuidanceRuleGroupFocusTarget({ guidanceRuleGroups, ruleItems }) {
  const requestedGroups = (guidanceRuleGroups ?? []).filter(Boolean)
  const items = Array.isArray(ruleItems) ? ruleItems : []

  for (const scope of requestedGroups) {
    const applicableItems = items
      .filter((item) => item?.scope === scope && item?.status !== 'na' && item?.rule_id)
      .sort((left, right) => getRuleStatusPriority(left.status) - getRuleStatusPriority(right.status))

    if (applicableItems.length > 0) {
      return applicableItems[0].rule_id
    }
  }

  return null
}

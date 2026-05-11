export interface GuidanceScopeRuleItem {
  rule_id: string
  status: string
  scope?: string
}

export interface ResolveGuidanceRuleGroupFocusTargetInput {
  guidanceRuleGroups?: string[]
  ruleItems?: GuidanceScopeRuleItem[]
}

export function resolveGuidanceRuleGroupFocusTarget(
  input: ResolveGuidanceRuleGroupFocusTargetInput,
): string | null

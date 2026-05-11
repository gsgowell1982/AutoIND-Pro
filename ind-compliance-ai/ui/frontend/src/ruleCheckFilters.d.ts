import type { RuleCheckItem } from './types'

export type RuleCheckStatusFilter = 'actionable' | 'fail' | 'warn' | 'pass' | 'na' | 'risk'

export interface RuleCheckStatusFilterOption {
  key: RuleCheckStatusFilter
  label: string
  count: number
  color: string
}

export interface RuleCheckStatusFilterSummary {
  hard_failures?: number
  soft_risks?: number
  pass_rules?: number
  na_rules?: number
}

export const DEFAULT_RULE_CHECK_STATUS_FILTER: RuleCheckStatusFilter

export function filterRuleCheckItems(
  items: RuleCheckItem[],
  filterKey?: RuleCheckStatusFilter | string,
  scopedRuleIds?: string[] | null,
): RuleCheckItem[]

export function getRuleCheckStatusFilterOptions(
  summary?: RuleCheckStatusFilterSummary,
): RuleCheckStatusFilterOption[]

export function getRuleCheckFilterLabel(filterKey?: RuleCheckStatusFilter | string): string

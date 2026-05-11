import type { RuleCheckDetails } from './types'

export interface RuleDetailDisplayRow {
  kind: 'text' | 'tag'
  label: string
  value: string
  color?: string
}

export function getRegulationSeverityLabel(severity?: string | null): string

export function getRegulationSeverityColor(severity?: string | null): string

export function getMatchStrengthLabel(matchStrength?: string | null): string | null

export function buildRequirementLabel(details?: RuleCheckDetails): string | null

export function buildRuleDetailDisplayRows(details?: RuleCheckDetails): RuleDetailDisplayRow[]

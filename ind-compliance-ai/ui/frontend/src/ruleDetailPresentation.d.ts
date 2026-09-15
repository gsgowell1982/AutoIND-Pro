import type { RuleCheckDetails } from './types'

export interface RuleDetailDisplayRow {
  kind: 'text' | 'tag'
  label: string
  value: string
  color?: string
}

export interface ViolationDetailDisplay {
  sectionIdentifier: string
  sequenceNumber: string
  previousSequenceNumber: string
  changedAttributes: string[]
  violationType: string
  message: string
  metadataChanges: Array<{
    attribute: string
    oldValue: string
    newValue: string
    changeType: string
  }>
  leafOperations: Array<{
    leafId: string
    operation: string
    filePath: string
  }>
}

export interface CrossModuleInconsistencyDisplay {
  sequenceNumber: string
  module1Section: string
  module2Section: string
  inconsistentAttributes: Array<{
    attribute: string
    value1: string
    value2: string
  }>
  message: string
}

export function getRegulationSeverityLabel(severity?: string | null): string

export function getRegulationSeverityColor(severity?: string | null): string

export function getMatchStrengthLabel(matchStrength?: string | null): string | null

export function buildRequirementLabel(details?: RuleCheckDetails): string | null

export function buildRuleDetailDisplayRows(details?: RuleCheckDetails): RuleDetailDisplayRow[]

export function buildEctdMetadataLifecycleDetails(details?: any): RuleDetailDisplayRow[]

export function buildViolationDetailDisplay(violation?: any): ViolationDetailDisplay

export function buildCrossModuleConsistencyDetails(details?: any): RuleDetailDisplayRow[]

export function buildCrossModuleInconsistencyDisplay(inconsistency?: any): CrossModuleInconsistencyDisplay


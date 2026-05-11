export function getRuleTitle(ruleId: string): string
export function buildRuleHeaderPresentation(ruleId: string): {
  title: string
  secondaryRuleId: string | null
}
export function getRuleSummary(ruleId: string, status: string, fallbackMessage?: string): string
export function getRuleStatusLabel(status: string): string
export function getRuleCategoryLabel(category: string): string
export function getRegulationSeverityLabel(severity: string): string
export function getRegulationSeverityColor(severity: string): string
export function getNavigationStatusLabel(status: string): string
export function getNavigationReasonLabel(reason: string): string
export function shouldDisplayNavigationDiagnostic(
  status: string | null | undefined,
  reason: string | null | undefined,
  jumpPage: number | null | undefined,
): boolean
export function buildRuleLocationLabelParts(document: {
  filename?: string | null
  module_label?: string | null
  section_refs?: Array<{
    outline_index?: string | null
    section_title?: string | null
  }>
}): string[]
export function getVerificationStatusLabel(status: string): string
export function getCitationLabel(citation: string): string
export function getRuleBasisPresentation(
  ruleId: string,
  citation: string,
): {
  basisKind: 'regulation' | 'system' | 'generic'
  basisLabel: string
}
export function buildRuleBasisDisplay(
  ruleId: string,
  citation: string,
  basis: {
    basis_kind?: 'regulation' | 'system' | 'generic' | string | null
    basis_label?: string | null
    basis_detail?: string | null
  } | null | undefined,
): {
  labelPrefix: string
  basisKind: 'regulation' | 'system' | 'generic' | string
  basisLabel: string
  basisDetail: string | null
} | null

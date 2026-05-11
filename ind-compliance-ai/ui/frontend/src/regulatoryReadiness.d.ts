export interface RegulatoryReadinessMatrixRow {
  key: string
  regulationId: string
  sourceFile: string
  coverageStatus: string
  coverageLabel: string
  statusColor: string
  clauseSummary: string
  ruleSummary: string
  productRoleLabel: string
  triage: string
  triageLabel: string
  triageColor: string
  automationBoundary: string
}

export function getRegulatoryReadinessCoverageLabel(status: string | null | undefined): string

export function getRegulatoryReadinessStatusColor(status: string | null | undefined): string

export function getRegulatoryReadinessTriageLabel(triage: string | null | undefined): string

export function getRegulatoryReadinessTriageColor(triage: string | null | undefined): string

export function getRegulatoryReadinessProductRoleLabel(role: string | null | undefined): string

export function buildRegulatoryReadinessMatrixRows(
  sources: import('./types').RegulatoryReadinessSource[] | null | undefined,
): RegulatoryReadinessMatrixRow[]

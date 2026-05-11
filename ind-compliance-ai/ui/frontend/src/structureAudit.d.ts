export interface StructureAuditFilterOptions {
  severityFilter?: 'warnings' | 'passes' | 'all'
  issueFilter?: string
  filenameFilter?: string
}

export interface StructureAuditIssueOption {
  value: string
  label: string
}

export interface StructureAuditIssueSummaryItem {
  issueKey: string
  label: string
  count: number
}

export interface StructureAuditNavigationTarget {
  key: 'toc' | 'body'
  label: string
  page: number
  tocSequenceId: string | null
}

export interface StructureAuditGroup {
  filename: string
  records: import('./types').RuleStructureAuditRecord[]
  warnCount: number
  passCount: number
  issueKeys: string[]
}

export function getStructureAuditIssueLabel(issueKey: string | null | undefined): string

export function getStructureAuditIssues(record: import('./types').RuleStructureAuditRecord | null | undefined): string[]

export function getStructureAuditSeverity(
  record: import('./types').RuleStructureAuditRecord | null | undefined,
): 'warn' | 'pass'

export function buildStructureAuditRecordSignature(
  records: import('./types').RuleStructureAuditRecord[] | null | undefined,
  options?: { pdfFileId?: string | null },
): string

export function filterStructureAuditRecords(
  records: import('./types').RuleStructureAuditRecord[] | null | undefined,
  options?: StructureAuditFilterOptions,
): import('./types').RuleStructureAuditRecord[]

export function buildStructureAuditGroups(
  records: import('./types').RuleStructureAuditRecord[] | null | undefined,
  options?: StructureAuditFilterOptions,
): StructureAuditGroup[]

export function getStructureAuditIssueOptions(): StructureAuditIssueOption[]

export function getStructureAuditFilenameOptions(records: import('./types').RuleStructureAuditRecord[] | null | undefined): StructureAuditIssueOption[]

export function buildStructureAuditIssueSummary(
  records: import('./types').RuleStructureAuditRecord[] | null | undefined,
  options?: StructureAuditFilterOptions,
): StructureAuditIssueSummaryItem[]

export function buildStructureAuditRuleFocusList<T extends { rule_id?: string | null }>(
  ruleItems: T[] | null | undefined,
  focusedRuleId: string | null | undefined,
): T[]

export function toggleStructureAuditFilterValue(
  currentValue: string | null | undefined,
  nextValue: string | null | undefined,
): string

export function buildStructureAuditNavigationTargets(
  record: import('./types').RuleStructureAuditRecord | null | undefined,
  options?: {
    activeFilename?: string | null
    tocSequences?: import('./types').PdfTocSequence[] | null | undefined
  },
): StructureAuditNavigationTarget[]

export function getStructureAuditRootPageConflictRows(
  record: import('./types').RuleStructureAuditRecord | null | undefined,
): import('./types').RuleStructureAuditRootPageAlignmentRow[]

export function getStructureAuditMissingPathRows(
  record: import('./types').RuleStructureAuditRecord | null | undefined,
  scope?: 'direct_child' | 'bounded_subtree',
): Array<{
  rootOutlineIndex: string | null
  parentOutlineIndex: string | null
  outlineIndex: string | null
  outlinePath: string | null
  textPath: string | null
  pageLocatorValue: number | null
  navigationPage: number | null
  level: number | null
  nearestBodyAnchorOutlineIndex: string | null
  nearestBodyAnchorPage: number | null
}>

export function getStructureAuditPathAlignmentRows(
  record: import('./types').RuleStructureAuditRecord | null | undefined,
): Array<{
  rootOutlineIndex: string | null
  parentOutlineIndex: string | null
  outlineIndex: string | null
  outlinePath: string | null
  textPath: string | null
  pageLocatorValue: number | null
  navigationPage: number | null
  level: number | null
  bodyAnchorOutlineIndex: string | null
  bodyAnchorPage: number | null
  bodyAnchorKind: string | null
  alignmentStatus: string | null
  pageOffset: number | null
  hasBodyMatch: boolean
}>

export function resolveStructureAuditPathNavigationTarget(
  pathRow:
    | {
        pageLocatorValue?: number | null
        navigationPage?: number | null
        nearestBodyAnchorPage?: number | null
        bodyAnchorPage?: number | null
      }
    | null
    | undefined,
  kind: 'toc' | 'body',
  options?: {
    tocBlocks?: Array<{ toc_id?: string | null; page?: number | null; toc_sequence_id?: string | null }> | null | undefined
    preferredTocSequenceId?: string | null
  },
): {
  page: number | null
  structuralId: string | null
  tocSequenceId: string | null
  navigationStatus: string
  navigationReason: string
}

export function buildStructureAuditPathFocusKey(
  pathRow:
    | {
        outlinePath?: string | null
        outlineIndex?: string | null
        pageLocatorValue?: number | null
      }
    | null
    | undefined,
  scope?: 'direct_child' | 'bounded_subtree' | 'toc_body',
): string

export function buildStructureAuditPathFocusLabel(
  pathRow:
    | {
        textPath?: string | null
        outlinePath?: string | null
        nearestBodyAnchorOutlineIndex?: string | null
      }
    | null
    | undefined,
  kind?: 'toc' | 'body',
): string

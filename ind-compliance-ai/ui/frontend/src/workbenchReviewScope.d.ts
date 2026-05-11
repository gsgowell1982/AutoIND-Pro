import type { WorkbenchPayload } from './types'

export type WorkbenchReviewScopeMode = 'empty' | 'single_file' | 'project' | 'controlled_demo' | 'unknown'

export interface WorkbenchReviewScope {
  mode: WorkbenchReviewScopeMode
  filename: string | null
  fileCount: number
  isControlledDemo: boolean
  isSingleFile: boolean
  isProjectReview: boolean
}

export interface ReviewPanelVisibility {
  scope: WorkbenchReviewScope
  showDemoPanels: boolean
  showSingleFileSummary: boolean
  showProjectReadinessPanels: boolean
  showStructureAudit: boolean
  showRuleChecks: boolean
}

export interface SingleFileReviewModule {
  name: string
  status: 'pass' | 'fail' | 'review_required' | 'not_available'
  summary: string
}

export interface SingleFileReviewSummary {
  title: string
  filename: string | null
  conclusion: string
  tags: Array<{ label: string; color: string }>
  modules: SingleFileReviewModule[]
}

export function resolveWorkbenchReviewScope(workbench: WorkbenchPayload | null): WorkbenchReviewScope

export function getReviewPanelVisibility(workbench: WorkbenchPayload | null): ReviewPanelVisibility

export function buildSingleFileReviewSummary(workbench: WorkbenchPayload | null): SingleFileReviewSummary

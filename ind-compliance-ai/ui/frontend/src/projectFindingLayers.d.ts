export interface ProjectFindingLayerCounts {
  fail: number
  review: number
  pass: number
  actionable?: number
}

export interface ProjectFindingLayer {
  id: 'structure_naming' | 'file_content' | 'cross_file_consistency'
  title: string
  description: string
  status: 'actionable' | 'clear' | 'not_available' | 'reserved'
  counts: ProjectFindingLayerCounts
  actionable: Array<Record<string, unknown>>
  passed: Array<Record<string, unknown>>
}

export interface ProjectFindingLayersResult {
  layers: ProjectFindingLayer[]
  crossFileConsistency: ProjectFindingLayer
  actionableCount: number
  passCount: number
  focusedPath: string | null
}

export function buildProjectFindingLayers(workbench: unknown, options?: { selectedPath?: string | null }): ProjectFindingLayersResult

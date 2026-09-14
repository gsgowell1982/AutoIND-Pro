export interface SelectedFileEvidenceMetrics {
  pageCount: number
  characterCount: number
  tableCount: number
  imageCount: number
}

export interface SelectedFileEvidence {
  kind: 'parsed' | 'support' | 'directory' | 'unavailable'
  document: Record<string, unknown> | null
  fileUrl: string | null
  preview: string
  message: string
  metrics: SelectedFileEvidenceMetrics
}

export function buildSelectedFileEvidence(input?: {
  selectedPath?: string | null
  documents?: Array<Record<string, unknown>>
  files?: Array<Record<string, unknown>>
  directoryPaths?: string[]
}): SelectedFileEvidence

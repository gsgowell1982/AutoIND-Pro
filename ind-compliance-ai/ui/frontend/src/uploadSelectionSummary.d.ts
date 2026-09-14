export interface UploadSelectionFileRecord {
  path?: string
  name?: string
  size?: number
}

export interface UploadSelectionSummaryInput {
  mode: string
  files?: UploadSelectionFileRecord[]
  directoryPaths?: string[]
}

export interface UploadSelectionSummary {
  mode: string
  fileCount: number
  directoryCount: number
  rootCount: number
  rootLabel: string | null
  totalBytes: number
}

export function buildUploadSelectionSummary(input: UploadSelectionSummaryInput): UploadSelectionSummary
export function filterUploadSelectionFiles(
  files: UploadSelectionFileRecord[] | undefined,
  query: string,
): UploadSelectionFileRecord[]

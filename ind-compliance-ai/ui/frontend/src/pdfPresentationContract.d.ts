export interface PdfPresentationContractSourceRow {
  key: string
  label: string
  filename: string
  section: string
  sections: string
  pageLabel: string
}

export interface PdfPresentationContractDisplay {
  schemaVersion: string
  sourceRows: PdfPresentationContractSourceRow[]
  deterministicLabels: string[]
  manualReviewItems: string[]
  hasManualReview: boolean
}

export function asContractRecord(value: unknown): Record<string, unknown> | null
export function buildPdfPresentationContractDisplay(
  contract?: unknown,
): PdfPresentationContractDisplay | null

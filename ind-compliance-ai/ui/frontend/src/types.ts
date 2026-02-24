export interface FileStatus {
  file_id: string
  filename: string
  status: string
  message?: string
  progress: number
}

export interface UploadJobResponse {
  job_id: string
  status: string
  files: FileStatus[]
}

export interface JobStatusResponse {
  job_id: string
  status: string
  progress: number
  created_at: string
  updated_at: string
  files: FileStatus[]
}

export interface BoundingBox {
  id: string
  page_number: number
  text: string
  bbox: {
    x0: number
    y0: number
    x1: number
    y1: number
  }
}

export interface PdfPageMeta {
  page_number: number
  width: number
  height: number
  text: string
  block_count: number
}

export interface PdfImageBlock {
  image_id: string
  page: number
  bbox: [number, number, number, number]
  title?: string
  figure_ref?: string
}

export interface PdfTableCell {
  row: number
  col: number
  text: string
  bbox: [number, number, number, number]
  rowspan?: number
  colspan?: number
}

export interface PdfTableAst {
  table_id: string
  page: number
  bbox: [number, number, number, number]
  header: Array<{ text: string; col: number }>
  cells: PdfTableCell[]
  column_hash?: string
  continued_from?: string
  continued_to?: string[]
  continuation_source?: {
    source_table_id: string
    strategy: string
    similarity: number
    hint_table_id?: string
    hint_similarity?: number
    inherited_fields?: string[]
  }
}

export interface PdfFigureRef {
  figure_ref: string
  image_id: string
  title?: string
  page: number
  bbox: [number, number, number, number]
}

export interface WorkbenchPayload {
  pdf_document: {
    file_id: string
    filename: string
    file_url: string
    pages: PdfPageMeta[]
    bounding_boxes: BoundingBox[]
    image_blocks?: PdfImageBlock[]
    table_asts?: PdfTableAst[]
    figures?: PdfFigureRef[]
  } | null
  markdown: string
  full_markdown_download_url?: string
  rule_checks: {
    enabled: boolean
    items: Array<{ id: string; message: string }>
    message: string
  }
}

export interface ConsistencyRow {
  fact: string
  module_values: Array<{ module: string; value: string }>
  is_consistent: boolean
}

export interface ConsistencyResponse {
  rows: ConsistencyRow[]
}

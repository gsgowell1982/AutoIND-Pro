const SOURCE_LABELS = {
  cn_technical_specification: '中国 eCTD 技术规范',
  ich_submission_formats: 'ICH PDF 提交格式规范',
  cn_validation_standard: '中国 eCTD 验证标准',
}

const CHECK_LABELS = {
  file_size: '文件大小',
  navigation: '长文档导航',
  hyperlink_integrity: '超链接完整性',
  validation_standard: 'PDF 验证标准元数据',
}

export function asContractRecord(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : null
}

export function buildPdfPresentationContractDisplay(contract) {
  const normalized = asContractRecord(contract)
  if (!normalized) return null

  const sourceReferences = asContractRecord(normalized.source_references) ?? {}
  const deterministicChecks = asContractRecord(normalized.deterministic_checks) ?? {}
  const boundary = asContractRecord(normalized.automation_boundary) ?? {}
  const sourceRows = Object.entries(sourceReferences).map(([key, value]) => {
    const source = asContractRecord(value) ?? {}
    const sections = Array.isArray(source.sections) ? source.sections.filter(Boolean).join(', ') : ''
    const pages = [source.pdf_page_start, source.pdf_page_end].filter((page) => Number(page) > 0)
    const pageLabel = pages.length === 2 ? `PDF ${pages[0]}-${pages[1]} 页` : Number(source.pdf_page) > 0 ? `PDF 第 ${source.pdf_page} 页` : ''
    return {
      key,
      label: SOURCE_LABELS[key] ?? key,
      filename: String(source.source_filename ?? '').trim(),
      section: String(source.section ?? '').trim(),
      sections,
      pageLabel,
    }
  })
  const deterministicLabels = Object.keys(deterministicChecks).map((key) => CHECK_LABELS[key] ?? key)
  const manualReviewItems = Array.isArray(boundary.manual_review)
    ? boundary.manual_review.map((item) => String(item).trim()).filter(Boolean)
    : []

  return {
    schemaVersion: String(normalized.schema_version ?? '').trim(),
    sourceRows,
    deterministicLabels,
    manualReviewItems,
    hasManualReview: manualReviewItems.length > 0,
  }
}

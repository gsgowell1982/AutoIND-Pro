function normalizePath(value) {
  return String(value ?? '').replaceAll('\\', '/').replace(/^\/+|\/+$/g, '').trim()
}

function asNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

export function buildSelectedFileEvidence({ selectedPath, documents = [], files = [], directoryPaths = [] } = {}) {
  const path = normalizePath(selectedPath)
  const document = documents.find((item) => normalizePath(item?.relative_path || item?.filename) === path) ?? null
  if (document) {
    return {
      kind: 'parsed',
      document,
      fileUrl: document.file_url || null,
      preview: String(document.text_preview || '').trim(),
      message: '该文件已完成解析，可查看文件级格式与内容证据。',
      metrics: {
        pageCount: asNumber(document.page_count),
        characterCount: asNumber(document.character_count),
        tableCount: asNumber(document.table_count),
        imageCount: asNumber(document.image_count),
      },
    }
  }

  const file = files.find((item) => normalizePath(item?.relative_path || item?.filename) === path) ?? null
  const supportMessage = String(file?.message || '').toLowerCase()
  if (file && (file.document_parse === false || supportMessage.includes('support file') || supportMessage.includes('skipped'))) {
    return {
      kind: 'support',
      document: null,
      fileUrl: file.file_url || null,
      preview: '',
      message: '这是 eCTD 支持文件，当前保留清单与结构证据，内容解析将在文件级格式/Schema 校验阶段执行。',
      metrics: { pageCount: 0, characterCount: 0, tableCount: 0, imageCount: 0 },
    }
  }

  const normalizedDirectories = directoryPaths.map(normalizePath).filter(Boolean)
  const isDirectory = normalizedDirectories.includes(path) || [...normalizedDirectories].some((item) => path.startsWith(`${item}/`))
  return {
    kind: isDirectory ? 'directory' : 'unavailable',
    document: null,
    fileUrl: file?.file_url || null,
    preview: '',
    message: isDirectory ? '已定位到目录，请选择具体文件查看文件级证据。' : '当前节点暂无可用的文件级解析证据。',
    metrics: { pageCount: 0, characterCount: 0, tableCount: 0, imageCount: 0 },
  }
}

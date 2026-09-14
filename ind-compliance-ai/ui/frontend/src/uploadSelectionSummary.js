function asArray(value) {
  return Array.isArray(value) ? value : []
}

function normalizePath(value) {
  return String(value ?? '')
    .replaceAll('\\', '/')
    .split('/')
    .filter(Boolean)
    .join('/')
}

function recordPath(record) {
  return normalizePath(record?.path || record?.name)
}

export function buildUploadSelectionSummary({ mode, files, directoryPaths }) {
  const normalizedFiles = asArray(files)
  const directories = new Set()
  const roots = new Set()

  asArray(directoryPaths).forEach((path) => {
    const normalizedPath = normalizePath(path)
    if (!normalizedPath) return
    directories.add(normalizedPath)
    roots.add(normalizedPath.split('/')[0])
  })

  normalizedFiles.forEach((file) => {
    const path = recordPath(file)
    if (!path) return
    const parts = path.split('/')
    roots.add(parts[0])
    parts.slice(0, -1).forEach((_, index) => directories.add(parts.slice(0, index + 1).join('/')))
  })

  const rootLabels = [...roots]

  return {
    mode,
    fileCount: normalizedFiles.length,
    directoryCount: directories.size,
    rootCount: rootLabels.length,
    rootLabel: rootLabels.length === 1 ? rootLabels[0] : null,
    totalBytes: normalizedFiles.reduce((total, file) => total + Math.max(0, Number(file?.size) || 0), 0),
  }
}

export function filterUploadSelectionFiles(files, query) {
  const normalizedQuery = normalizePath(query).toLowerCase()
  if (!normalizedQuery) return asArray(files)
  return asArray(files).filter((file) => recordPath(file).toLowerCase().includes(normalizedQuery))
}

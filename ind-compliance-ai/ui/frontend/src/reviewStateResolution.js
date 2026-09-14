export function resolveCurrentPage(pageOptions, currentPage) {
  const pages = Array.isArray(pageOptions)
    ? pageOptions.filter((page) => Number.isFinite(page) && page > 0)
    : []
  if (pages.includes(currentPage)) return currentPage
  return pages[0] ?? 1
}

export function resolveSelectedTocSequenceId({ sequenceIds, activePageSequenceIds, selectedId }) {
  const availableIds = Array.isArray(sequenceIds) ? sequenceIds.filter(Boolean) : []
  if (availableIds.length === 0) return null
  if (selectedId && availableIds.includes(selectedId)) return selectedId
  const activeIds = Array.isArray(activePageSequenceIds) ? activePageSequenceIds : []
  return activeIds.find((id) => availableIds.includes(id)) ?? availableIds[0]
}

export function resolveRequestedStructuralId(structuralId) {
  return typeof structuralId === 'string' && structuralId.trim() ? structuralId.trim() : null
}

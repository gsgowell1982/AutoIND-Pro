const ACTIONABLE_STATUSES = new Set(['fail', 'warn', 'warning', 'review_required', 'manual_review', 'pending_review'])
const PASS_STATUSES = new Set(['pass', 'passed', 'success', 'ok'])
const IGNORED_STATUSES = new Set(['na', 'not_applicable', 'not_available'])

function asArray(value) {
  return Array.isArray(value) ? value : []
}

function normalizePath(value) {
  return String(value ?? '').replaceAll('\\', '/').replace(/^\/+|\/+$/g, '').trim().toLowerCase()
}

function findingPath(item) {
  const direct = normalizePath(item?.relative_path || item?.path || item?.filename)
  if (direct) return direct
  const details = item?.details && typeof item.details === 'object' ? item.details : null
  const matched = asArray(details?.matched_documents)
  return normalizePath(matched[0]?.relative_path || matched[0]?.filename)
}

function filterFindingsByPath(items, selectedPath) {
  const selected = normalizePath(selectedPath)
  if (!selected) return asArray(items)
  return asArray(items).filter((item) => {
    const path = findingPath(item)
    if (!path) return true
    return path === selected || path.startsWith(`${selected}/`) || selected.startsWith(`${path}/`)
  })
}

function normalizeFindingStatus(status) {
  const normalized = String(status ?? '').trim().toLowerCase()
  if (PASS_STATUSES.has(normalized)) return 'pass'
  if (IGNORED_STATUSES.has(normalized)) return 'ignored'
  if (ACTIONABLE_STATUSES.has(normalized)) return normalized === 'fail' ? 'fail' : 'review'
  return 'review'
}

function classifyFindings(items, available = true) {
  const actionable = []
  const passed = []
  const counts = { fail: 0, review: 0, pass: 0 }

  asArray(items).forEach((item) => {
    const normalizedStatus = normalizeFindingStatus(item?.status)
    if (normalizedStatus === 'ignored') return
    const finding = { ...item, normalized_status: normalizedStatus }
    if (normalizedStatus === 'pass') {
      counts.pass += 1
      passed.push(finding)
      return
    }
    counts[normalizedStatus] += 1
    actionable.push(finding)
  })

  return {
    counts,
    actionable,
    passed,
    status: actionable.length > 0 ? 'actionable' : available ? 'clear' : 'not_available',
  }
}

function buildLayer(id, title, description, items, extras = {}) {
  const classified = classifyFindings(items, Array.isArray(items))
  return {
    id,
    title,
    description,
    ...classified,
    ...extras,
  }
}

export function buildProjectFindingLayers(workbench, { selectedPath = null } = {}) {
  const layers = [
    buildLayer(
      'structure_naming',
      '层级、命名与应有文件',
      '检查应用根、序列、模块、区域目录、文件位置与必需文件。',
      filterFindingsByPath(workbench?.package_findings, selectedPath),
    ),
    buildLayer(
      'file_content',
      '单文件内容合规',
      '检查已进入解析链路的文件格式、内容规则与证据边界；未解析的工具文件不被误报为通过。',
      filterFindingsByPath(workbench?.rule_checks?.items, selectedPath),
    ),
  ]

  const crossFileConsistency = {
    id: 'cross_file_consistency',
    title: '多文件一致性与逻辑',
    description: '跨文件字段一致性、引用关系和生命周期逻辑检查接口已预留。',
    status: 'reserved',
    counts: { fail: 0, review: 0, pass: 0, actionable: 0 },
    actionable: [],
    passed: [],
  }

  return {
    layers,
    crossFileConsistency,
    focusedPath: normalizePath(selectedPath) || null,
    actionableCount: layers.reduce((total, layer) => total + layer.actionable.length, 0),
    passCount: layers.reduce((total, layer) => total + layer.passed.length, 0),
  }
}

const ISSUE_LABELS = {
  root_coverage_gap: '根章节覆盖不足',
  root_page_order_conflict: '根章节页序冲突',
  root_page_offset_conflict: '根章节页码偏移冲突',
  root_page_span_conflict: '根章节页区间冲突',
  direct_child_coverage_gap: '直接子级覆盖不足',
  bounded_subtree_coverage_gap: '子树覆盖不足',
}

const ISSUE_ORDER = [
  'root_coverage_gap',
  'root_page_order_conflict',
  'root_page_offset_conflict',
  'root_page_span_conflict',
  'direct_child_coverage_gap',
  'bounded_subtree_coverage_gap',
]

function getIssueSortIndex(issueKey) {
  const index = ISSUE_ORDER.indexOf(issueKey)
  return index === -1 ? ISSUE_ORDER.length : index
}

export function getStructureAuditIssueLabel(issueKey) {
  return ISSUE_LABELS[issueKey] ?? issueKey ?? ''
}

export function getStructureAuditIssues(record) {
  const issues = []
  if (typeof record?.root_outline_coverage_ratio === 'number' && record.root_outline_coverage_ratio < 1) {
    issues.push('root_coverage_gap')
  }
  if (record?.root_page_order_ready === false) {
    issues.push('root_page_order_conflict')
  }
  if (record?.root_page_offset_ready === false) {
    issues.push('root_page_offset_conflict')
  }
  if (record?.root_page_span_ready === false) {
    issues.push('root_page_span_conflict')
  }
  if (typeof record?.direct_child_coverage_ratio === 'number' && record.direct_child_coverage_ratio < 1) {
    issues.push('direct_child_coverage_gap')
  }
  if (
    typeof record?.bounded_subtree_coverage_ratio === 'number' &&
    record.bounded_subtree_coverage_ratio < 1
  ) {
    issues.push('bounded_subtree_coverage_gap')
  }
  return issues
}

export function getStructureAuditSeverity(record) {
  return record?.alignment_ready ? 'pass' : 'warn'
}

export function buildStructureAuditRecordSignature(records, { pdfFileId = null } = {}) {
  const normalizedRecords = Array.isArray(records) ? records : []
  return [
    String(pdfFileId ?? 'no-pdf'),
    normalizedRecords.length,
    ...normalizedRecords.map((record) =>
      [
        String(record?.audit_record_id ?? ''),
        String(record?.filename ?? ''),
        getStructureAuditSeverity(record),
        Array.isArray(record?.toc_body_alignment_path_rows) ? record.toc_body_alignment_path_rows.length : 0,
      ].join(':'),
    ),
  ].join('|')
}

export function filterStructureAuditRecords(
  records,
  { severityFilter = 'all', issueFilter = 'all', filenameFilter = 'all' } = {},
) {
  const normalizedRecords = Array.isArray(records) ? records : []
  return normalizedRecords.filter((record) => {
    const severity = getStructureAuditSeverity(record)
    const issues = getStructureAuditIssues(record)

    if (severityFilter === 'warnings' && severity !== 'warn') {
      return false
    }
    if (severityFilter === 'passes' && severity !== 'pass') {
      return false
    }
    if (issueFilter !== 'all' && !issues.includes(issueFilter)) {
      return false
    }
    if (filenameFilter !== 'all' && String(record?.filename ?? 'unknown-document') !== filenameFilter) {
      return false
    }
    return true
  })
}

export function buildStructureAuditGroups(
  records,
  { severityFilter = 'all', issueFilter = 'all', filenameFilter = 'all' } = {},
) {
  const filteredRecords = filterStructureAuditRecords(records, {
    severityFilter,
    issueFilter,
    filenameFilter,
  }).slice()
  filteredRecords.sort((left, right) => {
    const filenameDelta = String(left?.filename ?? '').localeCompare(String(right?.filename ?? ''))
    if (filenameDelta !== 0) {
      return filenameDelta
    }

    const severityRankDelta =
      (getStructureAuditSeverity(left) === 'warn' ? 0 : 1) -
      (getStructureAuditSeverity(right) === 'warn' ? 0 : 1)
    if (severityRankDelta !== 0) {
      return severityRankDelta
    }

    const issueCountDelta = getStructureAuditIssues(right).length - getStructureAuditIssues(left).length
    if (issueCountDelta !== 0) {
      return issueCountDelta
    }

    const ruleIdDelta = String(left?.rule_id ?? '').localeCompare(String(right?.rule_id ?? ''))
    if (ruleIdDelta !== 0) {
      return ruleIdDelta
    }

    return String(left?.audit_record_id ?? '').localeCompare(String(right?.audit_record_id ?? ''))
  })

  const groups = []
  const groupMap = new Map()
  for (const record of filteredRecords) {
    const filename = String(record?.filename ?? 'unknown-document')
    if (!groupMap.has(filename)) {
      const group = {
        filename,
        records: [],
        warnCount: 0,
        passCount: 0,
        issueKeys: [],
      }
      groupMap.set(filename, group)
      groups.push(group)
    }
    const group = groupMap.get(filename)
    group.records.push(record)
    if (getStructureAuditSeverity(record) === 'warn') {
      group.warnCount += 1
    } else {
      group.passCount += 1
    }
    for (const issueKey of getStructureAuditIssues(record)) {
      if (!group.issueKeys.includes(issueKey)) {
        group.issueKeys.push(issueKey)
      }
    }
    group.issueKeys.sort((left, right) => getIssueSortIndex(left) - getIssueSortIndex(right))
  }

  return groups
}

export function getStructureAuditIssueOptions() {
  return [
    { value: 'all', label: '全部问题' },
    ...ISSUE_ORDER.map((value) => ({
      value,
      label: getStructureAuditIssueLabel(value),
    })),
  ]
}

export function getStructureAuditFilenameOptions(records) {
  const normalizedRecords = Array.isArray(records) ? records : []
  const counts = new Map()

  for (const record of normalizedRecords) {
    const filename = String(record?.filename ?? 'unknown-document')
    counts.set(filename, (counts.get(filename) ?? 0) + 1)
  }

  return [
    { value: 'all', label: '全部文件' },
    ...Array.from(counts.entries())
      .sort((left, right) => left[0].localeCompare(right[0]))
      .map(([filename, count]) => ({
        value: filename,
        label: `${filename} (${count})`,
      })),
  ]
}

export function buildStructureAuditIssueSummary(
  records,
  { severityFilter = 'all', issueFilter = 'all', filenameFilter = 'all' } = {},
) {
  const filteredRecords = filterStructureAuditRecords(records, {
    severityFilter,
    issueFilter,
    filenameFilter,
  })
  const issueCounts = new Map()

  for (const record of filteredRecords) {
    for (const issueKey of getStructureAuditIssues(record)) {
      issueCounts.set(issueKey, (issueCounts.get(issueKey) ?? 0) + 1)
    }
  }

  return Array.from(issueCounts.entries())
    .sort((left, right) => {
      const countDelta = right[1] - left[1]
      if (countDelta !== 0) {
        return countDelta
      }
      return getIssueSortIndex(left[0]) - getIssueSortIndex(right[0])
    })
    .map(([issueKey, count]) => ({
      issueKey,
      label: getStructureAuditIssueLabel(issueKey),
      count,
    }))
}

export function buildStructureAuditRuleFocusList(ruleItems, focusedRuleId) {
  const normalizedItems = Array.isArray(ruleItems) ? ruleItems.slice() : []
  if (!focusedRuleId) {
    return normalizedItems
  }

  const focusedItems = []
  const remainingItems = []
  for (const item of normalizedItems) {
    if (item?.rule_id === focusedRuleId) {
      focusedItems.push(item)
    } else {
      remainingItems.push(item)
    }
  }

  if (focusedItems.length === 0) {
    return normalizedItems
  }
  return [...focusedItems, ...remainingItems]
}

export function toggleStructureAuditFilterValue(currentValue, nextValue) {
  const normalizedCurrent = currentValue ?? 'all'
  const normalizedNext = nextValue ?? 'all'
  if (normalizedCurrent === normalizedNext) {
    return 'all'
  }
  return normalizedNext
}

export function buildStructureAuditNavigationTargets(
  record,
  { activeFilename = null, tocSequences = [] } = {},
) {
  const recordFilename = String(record?.filename ?? 'unknown-document')
  if (activeFilename && recordFilename !== activeFilename) {
    return []
  }

  const targets = []
  const normalizedSequences = Array.isArray(tocSequences) ? tocSequences : []
  const firstSequence = normalizedSequences[0] ?? null
  const firstTocPage =
    firstSequence?.pages?.find((page) => typeof page === 'number' && page > 0) ??
    (Array.isArray(firstSequence?.page_span) ? firstSequence.page_span[0] : null)

  if (typeof firstTocPage === 'number' && firstTocPage > 0) {
    targets.push({
      key: 'toc',
      label: '定位目录页',
      page: firstTocPage,
      tocSequenceId: firstSequence?.toc_sequence_id ?? null,
    })
  }

  const bodyPages = Array.isArray(record?.projected_root_page_values)
    ? record.projected_root_page_values.filter((page) => typeof page === 'number' && page > 0)
    : []
  if (bodyPages.length > 0) {
    targets.push({
      key: 'body',
      label: '定位正文根章节页',
      page: Math.min(...bodyPages),
      tocSequenceId: null,
    })
  }

  return targets
}

export function getStructureAuditRootPageConflictRows(record) {
  const alignmentRows = Array.isArray(record?.root_page_alignment_rows) ? record.root_page_alignment_rows : []
  return alignmentRows.filter(
    (row) => row?.order_conflict === true || row?.offset_conflict === true || row?.span_conflict === true,
  )
}

export function getStructureAuditMissingPathRows(record, scope = 'direct_child') {
  const sourceRows =
    scope === 'bounded_subtree'
      ? record?.missing_body_bounded_subtree_path_rows
      : record?.missing_body_direct_child_path_rows
  const normalizedRows = Array.isArray(sourceRows) ? sourceRows : []

  return normalizedRows
    .map((row) => ({
      rootOutlineIndex: String(row?.root_outline_index ?? '').trim() || null,
      parentOutlineIndex: String(row?.parent_outline_index ?? '').trim() || null,
      outlineIndex: String(row?.outline_index ?? '').trim() || null,
      outlinePath: String(row?.outline_path ?? '').trim() || null,
      textPath: String(row?.text_path ?? '').trim() || null,
      pageLocatorValue: typeof row?.page_locator_value === 'number' ? row.page_locator_value : null,
      navigationPage: typeof row?.navigation_page === 'number' ? row.navigation_page : null,
      level: typeof row?.level === 'number' ? row.level : null,
      nearestBodyAnchorOutlineIndex:
        String(row?.nearest_body_anchor_outline_index ?? '').trim() || null,
      nearestBodyAnchorPage: typeof row?.nearest_body_anchor_page === 'number' ? row.nearest_body_anchor_page : null,
    }))
    .filter((row) => row.outlineIndex || row.outlinePath || row.textPath)
}

export function getStructureAuditPathAlignmentRows(record) {
  const sourceRows = Array.isArray(record?.toc_body_alignment_path_rows)
    ? record.toc_body_alignment_path_rows
    : []

  return sourceRows
    .map((row) => ({
      rootOutlineIndex: String(row?.root_outline_index ?? '').trim() || null,
      parentOutlineIndex: String(row?.parent_outline_index ?? '').trim() || null,
      outlineIndex: String(row?.outline_index ?? '').trim() || null,
      outlinePath: String(row?.outline_path ?? '').trim() || null,
      textPath: String(row?.text_path ?? '').trim() || null,
      pageLocatorValue: typeof row?.page_locator_value === 'number' ? row.page_locator_value : null,
      navigationPage: typeof row?.navigation_page === 'number' ? row.navigation_page : null,
      level: typeof row?.level === 'number' ? row.level : null,
      bodyAnchorOutlineIndex: String(row?.body_anchor_outline_index ?? '').trim() || null,
      bodyAnchorPage: typeof row?.body_anchor_page === 'number' ? row.body_anchor_page : null,
      bodyAnchorKind: String(row?.body_anchor_kind ?? '').trim() || null,
      alignmentStatus: String(row?.alignment_status ?? '').trim() || null,
      pageOffset: typeof row?.page_offset === 'number' ? row.page_offset : null,
      hasBodyMatch: row?.has_body_match === true,
    }))
    .filter((row) => row.outlineIndex || row.outlinePath || row.textPath)
}

export function resolveStructureAuditPathNavigationTarget(
  pathRow,
  kind,
  { tocBlocks = [], preferredTocSequenceId = null } = {},
) {
  if (kind === 'toc') {
    const page =
      typeof pathRow?.navigationPage === 'number'
        ? pathRow.navigationPage
        : typeof pathRow?.pageLocatorValue === 'number'
          ? pathRow.pageLocatorValue
          : null
    if (typeof page !== 'number' || page <= 0) {
      return {
        page: null,
        structuralId: null,
        tocSequenceId: preferredTocSequenceId ?? null,
        navigationStatus: 'unresolved',
        navigationReason: 'missing_structure_audit_toc_target',
      }
    }
    const normalizedTocBlocks = Array.isArray(tocBlocks) ? tocBlocks : []
    const matchingBlock =
      normalizedTocBlocks.find(
        (tocBlock) =>
          tocBlock?.page === page &&
          (!preferredTocSequenceId || tocBlock?.toc_sequence_id === preferredTocSequenceId),
      ) ??
      normalizedTocBlocks.find((tocBlock) => tocBlock?.page === page) ??
      null
    return {
      page,
      structuralId: matchingBlock?.toc_id ?? null,
      tocSequenceId: matchingBlock?.toc_sequence_id ?? preferredTocSequenceId ?? null,
      navigationStatus: matchingBlock?.toc_id ? 'resolved_structural' : 'resolved_page',
      navigationReason: matchingBlock?.toc_id
        ? 'resolved_from_structure_audit_toc_path'
        : 'resolved_from_structure_audit_toc_page',
    }
  }

  const page =
    typeof pathRow?.bodyAnchorPage === 'number'
      ? pathRow.bodyAnchorPage
      : typeof pathRow?.nearestBodyAnchorPage === 'number'
        ? pathRow.nearestBodyAnchorPage
        : null
  return {
    page,
    structuralId: null,
    tocSequenceId: null,
    navigationStatus: typeof page === 'number' && page > 0 ? 'resolved_page' : 'unresolved',
      navigationReason:
      typeof page === 'number' && page > 0
        ? 'resolved_from_structure_audit_body_anchor'
        : 'missing_structure_audit_body_anchor',
  }
}

export function buildStructureAuditPathFocusKey(pathRow, scope = 'direct_child') {
  const outlinePath = String(pathRow?.outlinePath ?? '').trim()
  const outlineIndex = String(pathRow?.outlineIndex ?? '').trim()
  const pageLocatorValue =
    typeof pathRow?.pageLocatorValue === 'number' ? String(pathRow.pageLocatorValue) : 'na'
  return `${scope}:${outlinePath || outlineIndex || 'unknown'}:${pageLocatorValue}`
}

export function buildStructureAuditPathFocusLabel(pathRow, kind = 'toc') {
  const primaryPath = String(pathRow?.textPath ?? '').trim() || String(pathRow?.outlinePath ?? '').trim() || '未解析路径'
  if (kind === 'body') {
    const anchor = String(pathRow?.nearestBodyAnchorOutlineIndex ?? '').trim()
    return anchor ? `${primaryPath} -> 正文锚点 ${anchor}` : `${primaryPath} -> 正文锚点`
  }
  return `${primaryPath} -> 目录路径`
}

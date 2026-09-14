function asArray(value) {
  return Array.isArray(value) ? value : []
}

function asNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function countRuleStatuses(items) {
  const counts = {
    fail: 0,
    warn: 0,
    pass: 0,
    na: 0,
  }
  for (const item of asArray(items)) {
    const status = String(item?.status ?? '')
    if (Object.prototype.hasOwnProperty.call(counts, status)) {
      counts[status] += 1
    }
  }
  return counts
}

function buildTocSequenceSelectionNote(structureRecords) {
  const selectedIds = []
  const excludedIds = []
  let excludedCount = 0

  for (const record of asArray(structureRecords)) {
    for (const sequenceId of asArray(record?.toc_sequence_ids)) {
      const normalized = String(sequenceId ?? '').trim()
      if (normalized && !selectedIds.includes(normalized)) {
        selectedIds.push(normalized)
      }
    }
    for (const sequenceId of asArray(record?.excluded_toc_sequence_ids)) {
      const normalized = String(sequenceId ?? '').trim()
      if (normalized && !excludedIds.includes(normalized)) {
        excludedIds.push(normalized)
      }
    }
    excludedCount += asNumber(record?.excluded_toc_sequence_count)
  }

  const effectiveExcludedCount = excludedIds.length || excludedCount
  if (selectedIds.length === 0 || effectiveExcludedCount === 0) {
    return ''
  }

  return `已按正文匹配证据选择主目录序列 ${selectedIds.join('、')}；其余 ${effectiveExcludedCount} 个目录序列未纳入当前正文一致性判定。`
}

function getUploadOverview(workbench) {
  return (
    workbench?.rule_checks?.summary?.upload_scope_overview ??
    workbench?.upload_scope_overview ??
    null
  )
}

export function resolveWorkbenchReviewScope(workbench) {
  if (!workbench) {
    return {
      mode: 'empty',
      filename: null,
      fileCount: 0,
      isControlledDemo: false,
      isSingleFile: false,
      isProjectReview: false,
    }
  }

  const demoSample = workbench.demo_sample ?? null
  const isControlledDemo = demoSample?.synthetic === true
  const pdfDocument = workbench.pdf_document ?? null
  const packageInventory = workbench.package_inventory ?? null
  const declaredReviewScope = String(workbench.review_scope ?? '').trim().toLowerCase()
  const uploadOverview = getUploadOverview(workbench)
  const overviewFileCount = asNumber(uploadOverview?.file_count)
  const inventoryFileCount = asArray(packageInventory?.file_paths).length
  const hasSinglePdf = Boolean(pdfDocument?.filename)
  const fileCount = overviewFileCount > 0
    ? overviewFileCount
    : inventoryFileCount > 0
      ? inventoryFileCount
      : hasSinglePdf
        ? 1
        : 0
  const isProjectReview = Boolean(packageInventory) || declaredReviewScope === 'sequence' || declaredReviewScope === 'application'

  if (isControlledDemo) {
    return {
      mode: 'controlled_demo',
      filename: null,
      fileCount,
      isControlledDemo: true,
      isSingleFile: false,
      isProjectReview: false,
    }
  }

  if (!isProjectReview && hasSinglePdf && fileCount <= 1) {
    return {
      mode: 'single_file',
      filename: pdfDocument.filename,
      fileCount: 1,
      isControlledDemo: false,
      isSingleFile: true,
      isProjectReview: false,
    }
  }

  return {
    mode: isProjectReview || fileCount > 1 ? 'project' : 'unknown',
    filename: pdfDocument?.filename ?? null,
    fileCount,
    isControlledDemo: false,
    isSingleFile: false,
    isProjectReview: isProjectReview || fileCount > 1,
  }
}

export function getReviewPanelVisibility(workbench) {
  const scope = resolveWorkbenchReviewScope(workbench)
  const hasRuleItems = asArray(workbench?.rule_checks?.items).length > 0
  const hasStructureAudit = asArray(workbench?.rule_checks?.structure_audit_records).length > 0

  return {
    scope,
    showDemoPanels: scope.mode === 'controlled_demo',
    showSingleFileSummary: scope.mode === 'single_file',
    showProjectReadinessPanels: scope.mode !== 'single_file',
    showStructureAudit: hasStructureAudit,
    showRuleChecks: Boolean(workbench?.rule_checks?.enabled) && hasRuleItems,
  }
}

export function buildSingleFileReviewSummary(workbench) {
  const scope = resolveWorkbenchReviewScope(workbench)
  const structureRecords = asArray(workbench?.rule_checks?.structure_audit_records)
  const structureWarningCount = structureRecords.filter((record) => record?.alignment_ready === false).length
  const structurePassCount = structureRecords.filter((record) => record?.alignment_ready === true).length
  const tocSequenceSelectionNote = buildTocSequenceSelectionNote(structureRecords)
  const ruleCounts = countRuleStatuses(workbench?.rule_checks?.items)
  const summaryCounts = workbench?.rule_checks?.summary ?? {}
  const hardFailures = asNumber(summaryCounts.hard_failures) || ruleCounts.fail
  const softRisks = asNumber(summaryCounts.soft_risks) || ruleCounts.warn
  const passRules = asNumber(summaryCounts.pass_rules) || ruleCounts.pass
  const prerequisiteCount =
    asNumber(workbench?.dossier_checklist?.summary?.prerequisite_required_count)

  const modules = []
  if (structureRecords.length > 0) {
    modules.push({
      name: '格式与结构',
      status: structureWarningCount > 0 ? 'review_required' : 'pass',
      summary:
        structureWarningCount > 0
          ? `发现 ${structureWarningCount} 条需要复核的结构定位或目录正文对应提示。${tocSequenceSelectionNote ? ` ${tocSequenceSelectionNote}` : ''}`
          : `已生成 ${structurePassCount || structureRecords.length} 条结构审计记录，未发现需优先提示的格式或结构风险。${tocSequenceSelectionNote ? ` ${tocSequenceSelectionNote}` : ''}`,
    })
  } else {
    modules.push({
      name: '格式与结构',
      status: 'not_available',
      summary: '当前文件未生成结构审计记录，可先查看左侧 PDF 与 Markdown 解析结果。',
    })
  }

  modules.push({
    name: '规则与前置条件',
    status: hardFailures > 0 ? 'fail' : softRisks > 0 ? 'review_required' : 'pass',
    summary:
      hardFailures > 0
        ? `发现 ${hardFailures} 条硬规则失败，需优先处理。`
        : softRisks > 0
          ? `发现 ${softRisks} 条规则风险或证据边界提示，建议结合原文人工复核。`
          : `已通过 ${passRules} 条规则提示；不适用规则不作为当前文件问题呈现。`,
  })

  const conclusion =
    hardFailures > 0 || structureWarningCount > 0
      ? '本次审阅对象为单个 PDF 文件，不按完整 IND 项目作结论；当前结果优先提示可定位到该文件的格式、结构、规则和前置条件线索。'
      : '本次审阅对象为单个 PDF 文件，不按完整 IND 项目作结论；当前未发现需要在此面板优先提示的格式、结构或规则风险。'

  return {
    title: '单个 PDF 文件审阅',
    filename: scope.filename,
    conclusion,
    tags: [
      { label: '单文件审阅', color: 'blue' },
      { label: structureRecords.length > 0 ? `结构记录 ${structureRecords.length}` : '暂无结构记录', color: 'cyan' },
      { label: `规则提示 ${asArray(workbench?.rule_checks?.items).length}`, color: 'green' },
      ...(prerequisiteCount > 0 ? [{ label: `需前置条件 ${prerequisiteCount}`, color: 'gold' }] : []),
    ],
    modules,
  }
}

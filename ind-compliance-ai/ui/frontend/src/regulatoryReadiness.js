const COVERAGE_LABELS = {
  closed: '已闭环',
  traceability_closed: '追溯闭环',
  parsed_not_coverage_closed: '已解析，未覆盖闭环',
}

const COVERAGE_COLORS = {
  closed: 'green',
  traceability_closed: 'cyan',
  parsed_not_coverage_closed: 'gold',
}

const TRIAGE_LABELS = {
  deterministic: '可确定性判定',
  prerequisite_required: '需要前置条件',
  human_review: '建议人工审核',
  out_of_scope_low_roi: '低性价比暂不实现',
}

const TRIAGE_COLORS = {
  deterministic: 'green',
  prerequisite_required: 'gold',
  human_review: 'orange',
  out_of_scope_low_roi: 'default',
}

const PRODUCT_ROLE_LABELS = {
  deterministic_validation_backbone: '确定性验证骨架',
  technical_spec_traceability_backbone: '技术规范追溯骨架',
  operational_guidance_and_explanation: '实施解释与操作提示',
  legal_background_risk_guidance: '法律背景风险提示',
  dossier_checklist_and_applicability_backbone: '资料清单与适用性骨架',
}

function positiveNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null
}

function buildClauseSummary(source) {
  const parts = [`条款 ${source?.clause_count ?? 0}`]
  const coveredCount = positiveNumber(source?.covered_count)
  const partialCount = positiveNumber(source?.partial_count)
  const deferredCount = positiveNumber(source?.deferred_count)
  const citationOnlyCount = positiveNumber(source?.citation_only_count)
  const requirementCount = positiveNumber(source?.requirement_count)
  const traceabilityGapCount = positiveNumber(source?.traceability_gap_count)

  if (coveredCount !== null) {
    parts.push(`已覆盖 ${coveredCount}`)
  }
  if (partialCount !== null) {
    parts.push(`部分 ${partialCount}`)
  }
  if (deferredCount !== null) {
    parts.push(`延后 ${deferredCount}`)
  }
  if (citationOnlyCount !== null) {
    parts.push(`引用/统计 ${citationOnlyCount}`)
  }
  if (requirementCount !== null) {
    parts.push(`需求 ${requirementCount}`)
  }
  if (traceabilityGapCount !== null) {
    parts.push(`追溯缺口 ${traceabilityGapCount}`)
  }
  return parts.join(' | ')
}

function buildRuleSummary(source) {
  const parts = [`候选 ${source?.rule_candidate_count ?? 0}`]
  const directIshCount = positiveNumber(source?.direct_ish_candidate_count)
  const directDraftCount = positiveNumber(source?.direct_rule_draft_count)

  if (directIshCount !== null) {
    parts.push(`直接倾向 ${directIshCount}`)
  }
  parts.push(`直接草案 ${directDraftCount ?? 0}`)
  return parts.join(' | ')
}

export function getRegulatoryReadinessCoverageLabel(status) {
  return COVERAGE_LABELS[status] ?? status ?? ''
}

export function getRegulatoryReadinessStatusColor(status) {
  return COVERAGE_COLORS[status] ?? 'default'
}

export function getRegulatoryReadinessTriageLabel(triage) {
  return TRIAGE_LABELS[triage] ?? triage ?? ''
}

export function getRegulatoryReadinessTriageColor(triage) {
  return TRIAGE_COLORS[triage] ?? 'default'
}

export function getRegulatoryReadinessProductRoleLabel(role) {
  return PRODUCT_ROLE_LABELS[role] ?? role ?? ''
}

export function buildRegulatoryReadinessMatrixRows(sources) {
  return (Array.isArray(sources) ? sources : []).map((source) => ({
    key: source?.regulation_id ?? source?.source_file ?? 'unknown-regulation-source',
    regulationId: source?.regulation_id ?? '',
    sourceFile: source?.source_file ?? '',
    coverageStatus: source?.coverage_status ?? '',
    coverageLabel: getRegulatoryReadinessCoverageLabel(source?.coverage_status),
    statusColor: getRegulatoryReadinessStatusColor(source?.coverage_status),
    clauseSummary: buildClauseSummary(source),
    ruleSummary: buildRuleSummary(source),
    productRoleLabel: getRegulatoryReadinessProductRoleLabel(source?.recommended_product_role),
    triage: source?.default_triage ?? '',
    triageLabel: getRegulatoryReadinessTriageLabel(source?.default_triage),
    triageColor: getRegulatoryReadinessTriageColor(source?.default_triage),
    automationBoundary: source?.automation_boundary ?? '',
  }))
}

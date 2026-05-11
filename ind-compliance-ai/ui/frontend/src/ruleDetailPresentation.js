const REGULATION_SEVERITY_LABELS = {
  错误: '错误',
  警告: '警告',
  提示信息: '提示信息',
}

const MATCH_STRENGTH_LABELS = {
  explicit_structure: '显式结构命中',
  weak_text_only: '弱文本信号',
  weak_text_llm_confirmed: '弱文本 + LLM 确认',
  not_applicable: '不适用',
  pdf_disallowed_hyperlink_action_detected: '检测到不允许的 PDF 超链接动作',
  pdf_link_action_whitelist_satisfied: 'PDF 超链接动作符合白名单',
  pdf_disallowed_bookmark_action_detected: '检测到不允许的 PDF 书签动作',
  pdf_bookmark_action_whitelist_satisfied: 'PDF 书签动作符合白名单',
}

export function getRegulationSeverityLabel(severity) {
  return REGULATION_SEVERITY_LABELS[severity] ?? severity ?? ''
}

export function getRegulationSeverityColor(severity) {
  switch (severity) {
    case '错误':
      return 'red'
    case '警告':
      return 'orange'
    case '提示信息':
      return 'blue'
    default:
      return 'default'
  }
}

export function getMatchStrengthLabel(matchStrength) {
  return MATCH_STRENGTH_LABELS[matchStrength] ?? null
}

export function buildRequirementLabel(details = {}) {
  const parts = [details.requirement_id, details.citation_anchor].filter((value) => Boolean(value))
  return parts.length > 0 ? parts.join(' | ') : null
}

export function buildRuleDetailDisplayRows(details = {}) {
  const rows = []
  const regulationSeverity = details.regulation_severity?.trim?.() || null
  const matchStrengthLabel = getMatchStrengthLabel(details.match_strength)

  if (regulationSeverity) {
    rows.push({
      kind: 'tag',
      label: '法规级别',
      value: getRegulationSeverityLabel(regulationSeverity),
      color: getRegulationSeverityColor(regulationSeverity),
    })
  }
  if (matchStrengthLabel) {
    rows.push({ kind: 'text', label: '命中强度', value: matchStrengthLabel })
  }

  return rows
}

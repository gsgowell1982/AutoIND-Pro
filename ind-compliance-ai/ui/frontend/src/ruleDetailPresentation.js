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

/**
 * 构建 eCTD 元数据生命周期耦合规则 (HR-ECTD-200) 的详细展示
 * @param {Object} details - 规则详细信息
 * @returns {Array} 展示行数组
 */
export function buildEctdMetadataLifecycleDetails(details = {}) {
  const rows = []

  // 序列信息
  if (details.sequence_number) {
    rows.push({
      kind: 'text',
      label: '当前序列',
      value: details.sequence_number,
    })
  }
  if (details.previous_sequence_number) {
    rows.push({
      kind: 'text',
      label: '前序列',
      value: details.previous_sequence_number,
    })
  }

  // 统计信息
  if (typeof details.total_sections_analyzed === 'number') {
    rows.push({
      kind: 'text',
      label: '分析的section总数',
      value: String(details.total_sections_analyzed),
    })
  }
  if (typeof details.violation_count === 'number') {
    rows.push({
      kind: 'tag',
      label: '违规数量',
      value: String(details.violation_count),
      color: details.violation_count > 0 ? 'red' : 'green',
    })
  }

  return rows
}

/**
 * 构建违规详情的展示结构
 * @param {Object} violation - 违规详情对象
 * @returns {Object} 结构化的违规信息
 */
export function buildViolationDetailDisplay(violation = {}) {
  return {
    sectionIdentifier: violation.section_identifier || '',
    sequenceNumber: violation.sequence_number || '',
    previousSequenceNumber: violation.previous_sequence_number || '',
    changedAttributes: violation.changed_attributes || [],
    violationType: violation.violation_type || '',
    message: violation.violation_message || violation.message || '',
    metadataChanges: (violation.metadata_changes || []).map(change => ({
      attribute: change.attribute || change.attribute_name || '',
      oldValue: change.old_value !== undefined ? String(change.old_value) : '',
      newValue: change.new_value !== undefined ? String(change.new_value) : '',
      changeType: change.change_type || '',
    })),
    leafOperations: (violation.leaf_operations || []).map(op => ({
      leafId: op.leaf_id || '',
      operation: op.operation || '',
      filePath: op.file_path || '',
    })),
  }
}

/**
 * 构建跨模块不一致 (HR-ECTD-201) 的详细展示
 * @param {Object} details - 规则详细信息
 * @returns {Array} 展示行数组
 */
export function buildCrossModuleConsistencyDetails(details = {}) {
  const rows = []

  // 序列信息
  if (details.sequence_number) {
    rows.push({
      kind: 'text',
      label: '序列号',
      value: details.sequence_number,
    })
  }

  // 统计信息
  if (typeof details.pairs_checked === 'number') {
    rows.push({
      kind: 'text',
      label: '检查的模块配对数',
      value: String(details.pairs_checked),
    })
  }
  if (typeof details.inconsistency_count === 'number') {
    rows.push({
      kind: 'tag',
      label: '不一致数量',
      value: String(details.inconsistency_count),
      color: details.inconsistency_count > 0 ? 'red' : 'green',
    })
  }

  return rows
}

/**
 * 构建跨模块不一致详情的展示结构
 * @param {Object} inconsistency - 不一致详情对象
 * @returns {Object} 结构化的不一致信息
 */
export function buildCrossModuleInconsistencyDisplay(inconsistency = {}) {
  const inconsistentAttrs = inconsistency.inconsistent_attributes || {}
  const formattedAttrs = Object.entries(inconsistentAttrs).map(([attr, values]) => {
    if (Array.isArray(values) && values.length === 2) {
      return {
        attribute: attr,
        value1: String(values[0]),
        value2: String(values[1]),
      }
    } else if (typeof values === 'object' && values !== null) {
      return {
        attribute: attr,
        value1: String(values.value1 || values[0] || ''),
        value2: String(values.value2 || values[1] || ''),
      }
    }
    return {
      attribute: attr,
      value1: '',
      value2: '',
    }
  })

  return {
    sequenceNumber: inconsistency.sequence_number || '',
    module1Section: inconsistency.module1_section || '',
    module2Section: inconsistency.module2_section || '',
    inconsistentAttributes: formattedAttrs,
    message: inconsistency.message || '',
  }
}

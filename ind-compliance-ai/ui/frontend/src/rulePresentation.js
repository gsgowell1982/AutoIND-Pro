const RULE_TITLES = {
  'HR-ECTD-045': 'PDF 超链接动作符合验证标准白名单',
  'HR-ECTD-001': 'eCTD 序列号递增校验',
  'HR-ECTD-041': 'PDF 文件不得包含嵌入附件',
  'SR-ECTD-017': 'PDF 文件不得包含非链接注释',
  'HR-PARSE-001': '解析结果具备规则判断所需证据',
  'HR-PARSE-002': '解析诊断未阻断自动规则判断',
  'HR-NAV-001': '目录与导航结构内部一致',
  'HR-CTD-001': '模块 3 / CMC 材料已配套质量综述 QOS',
  'SR-CTD-002': '当前材料集具备 CTD 基础申报结构信号',
  'SR-REG-006': '材料证据链支撑《实施条例》第六条的数据可追溯要求',
  'SR-FACT-001': '跨文档原子事实保持一致',
  'SR-EVID-001': '表格与图片已投影为可审阅证据',
  'SR-STRUCT-001': '章节与模块锚点可供下游规则消费',
  'SR-TOC-001': '目录与正文章节树主干存在可验证重合',
  'SR-FACT-002': '原子事实保留到内容单元级出处',
  'SR-FACT-003': '单元级事实信号与归一化事实一致',
  'SR-FACT-004': '同一文档内重复事实信号保持一致',
  'SR-FACT-005': '原子事实由内容型单元支撑而非仅上下文支撑',
}

const RULE_SUMMARY_BY_STATUS = {
  'HR-ECTD-001': {
    pass: '已验证 eCTD 序列号格式与包内目录一致性。',
    fail: '检测到 eCTD 序列号格式异常、与包内目录不一致，或与已知历史序列不一致。',
    na: '未检测到 eCTD 序列号元数据，当前不适用序列号递增检查。',
  },
  'HR-ECTD-045': {
    pass: 'PDF 超链接动作均在验证标准允许范围内。',
    fail: '检测到 PDF 超链接动作不在验证标准允许范围内。',
    na: '当前材料未检测到需要校验的 PDF 超链接动作。',
  },
  'HR-ECTD-041': {
    pass: '已检查适用 PDF 文件，未发现嵌入附件。',
    fail: '检测到 PDF 文件包含嵌入附件，不符合验证标准对附件的限制。',
    na: '当前未检测到适用的 PDF 文件，本规则不适用。',
  },
  'SR-ECTD-017': {
    pass: '已检查适用 PDF 文件，未发现非链接注释。',
    warn: '检测到 PDF 文件包含非链接注释，建议复核是否符合验证标准限制。',
    na: '当前未检测到适用的 PDF 文件，本规则不适用。',
  },
  'HR-PARSE-001': {
    pass: '已确认所有解析文档都具备规则判断所需的证据与内容单元。',
    fail: '存在文档缺少规则判断所需的证据或内容单元。',
  },
  'HR-PARSE-002': {
    pass: '未发现会阻断规则判断的解析 review-required 诊断项。',
    fail: '存在 parser review-required 诊断项，当前不宜直接信任自动规则结果。',
  },
  'HR-NAV-001': {
    pass: '目录与导航元数据当前保持一致。',
    fail: '目录与导航元数据存在不一致，需要先修复再信任自动导航。',
    na: '当前未检测到目录/导航结构，本规则不适用。',
  },
  'HR-CTD-001': {
    pass: '已识别到模块 3 / CMC 材料，并检测到质量综述/QOS 文件。',
    fail: '已识别到模块 3 / CMC 材料，但未检测到质量综述/QOS 文件。',
    na: '当前材料未出现明确的模块 3 / CMC 信号，本规则不适用。',
  },
  'SR-CTD-002': {
    pass: '已识别到足以支撑 CTD 基础申报结构的材料信号。',
    warn: '仅检测到较弱的 CTD 结构信号，建议人工复核。',
    na: '当前材料未检测到可消费的 CTD 基础申报结构信号。',
  },
  'SR-REG-006': {
    pass: '当前材料证据链可以支撑《实施条例》第六条关于数据真实、准确、完整、可追溯的自动化判断。',
    warn: '当前材料在数据可追溯或一致性方面存在自动化风险信号，建议人工复核。',
    na: '当前材料缺少足以支撑《实施条例》第六条自动判断的事实证据链。',
  },
  'SR-FACT-001': {
    pass: '当前可比对的跨文档原子事实保持一致。',
    warn: '存在跨文档原子事实不一致，建议重点复核一致性。',
    na: '当前缺少足够的跨文档可比事实，本规则不适用。',
  },
  'SR-EVID-001': {
    pass: '已确认表格与图片投影为可审阅证据与内容单元。',
    warn: '存在表格或图片未完整投影为可审阅证据。',
    na: '当前没有需要验证的非文本资产，本规则不适用。',
  },
  'SR-STRUCT-001': {
    pass: '章节与模块锚点当前可供下游规则消费。',
    warn: '章节与模块锚点仍有缺口，可能影响下游规则判断。',
    na: '当前材料缺少可消费的章节或模块锚点，本规则不适用。',
  },
  'SR-TOC-001': {
    pass: '目录根章节与正文章节树主干当前存在可验证重合，可用于后续目录一致性判断。',
    warn: '目录根章节与正文章节树主干缺少可验证重合，建议先复核目录或结构锚点。',
    na: '当前没有可用于比对的目录章节号，本规则不适用。',
  },
  'SR-FACT-002': {
    pass: '已提取事实当前保留到内容单元级出处。',
    warn: '存在事实缺少单元级出处，建议人工复核。',
    na: '当前没有提取到需要校验出处的原子事实。',
  },
  'SR-FACT-003': {
    pass: '单元级事实信号与归一化事实当前一致。',
    warn: '存在单元级事实信号与归一化事实不一致。',
    na: '当前没有足够的单元级事实信号，本规则不适用。',
  },
  'SR-FACT-004': {
    pass: '同一文档内重复事实信号当前一致。',
    warn: '同一文档内存在重复事实信号冲突。',
    na: '当前没有足够的重复事实信号，本规则不适用。',
  },
  'SR-FACT-005': {
    pass: '原子事实当前由内容型单元支撑。',
    warn: '存在仅由上下文型单元支撑的原子事实。',
    na: '当前没有可用于支撑质量校验的原子事实。',
  },
}

const STATUS_LABELS = {
  pass: '通过',
  warn: '预警',
  fail: '失败',
  na: '不适用',
}

const CATEGORY_LABELS = {
  hard: '硬规则',
  soft: '软规则',
}

const REGULATION_SEVERITY_LABELS = {
  错误: '错误',
  警告: '警告',
  提示信息: '提示信息',
}

const NAVIGATION_STATUS_LABELS = {
  resolved_structural: '结构目标已解析',
  resolved_page: '页级目标已解析',
  unresolved: '导航目标未解析',
}

const NAVIGATION_REASON_LABELS = {
  resolved_from_structural_evidence: '已从结构化证据解析到目标结构',
  resolved_from_text_evidence: '已从文本证据解析到目标页',
  fallback_to_section_anchor: '通过章节锚点回退到目标页',
  fallback_to_snippet_page: '通过片段所在页回退到目标页',
  no_navigation_target: '当前没有可用的确定性导航目标',
}

const VERIFICATION_STATUS_LABELS = {
  verified_structural: '端到端验证通过（结构级）',
  verified_page: '端到端验证通过（页级）',
  verification_waiting: '端到端验证等待中',
  verification_failed: '端到端验证失败',
  verification_degraded: '端到端验证降级',
  verification_unavailable: '端到端验证不可用',
}

const CITATION_LABELS = {
  'material-review-contract-v1#documents.summary': '材料审阅契约 v1 / 文档摘要',
  'material-review-contract-v1#diagnostic_index': '材料审阅契约 v1 / 解析诊断索引',
  'material-review-contract-v1#navigation_index': '材料审阅契约 v1 / 导航索引',
  'material-review-contract-v1#documents.classification': '材料审阅契约 v1 / 文档分类',
  'material-review-contract-v1#fact_index': '材料审阅契约 v1 / 原子事实索引',
  'material-review-contract-v1#evidence_index': '材料审阅契约 v1 / 证据索引',
  'material-review-contract-v1#section_index': '材料审阅契约 v1 / 章节锚点索引',
  'material-review-contract-v1#section_tree': '材料审阅契约 v1 / 章节树',
  'material-review-contract-v1#fact_signal_index': '材料审阅契约 v1 / 事实信号索引',
  'cn_ectd_technical_specification#sec_2_3_1': 'eCTD 技术规范 2.3.1 序列号',
  'cn_drug_registration_classification_and_dossier_requirements#art_015':
    '《药品注册分类及申报资料要求》 第三部分（申报资料要求）第（一）项',
}

export function getRuleTitle(ruleId) {
  return RULE_TITLES[ruleId] ?? ruleId
}

export function buildRuleHeaderPresentation(ruleId) {
  const title = getRuleTitle(ruleId)
  return {
    title,
    secondaryRuleId: title === ruleId ? null : ruleId,
  }
}

export function getRuleSummary(ruleId, status, fallbackMessage) {
  const summary = RULE_SUMMARY_BY_STATUS[ruleId]?.[status]
  if (summary) {
    return applyRuleSummaryCount(summary, fallbackMessage)
  }
  return localizeRuleFallbackSummary(status, fallbackMessage)
}

function applyRuleSummaryCount(summary, fallbackMessage) {
  const count = extractApplicableDocumentCount(fallbackMessage)
  if (!count) {
    return summary
  }
  return summary.replace('适用 PDF 文件', ` ${count} 个适用 PDF 文件`)
}

function extractApplicableDocumentCount(message) {
  const match = String(message ?? '').match(/\b(\d+)\s+applicable\s+PDF\s+document\(s\)/i)
  return match?.[1] ?? ''
}

function isProbablyEnglish(value) {
  return /[A-Za-z]{3,}/.test(String(value ?? ''))
}

function localizeRuleFallbackSummary(status, fallbackMessage) {
  const fallback = String(fallbackMessage ?? '').trim()
  if (!fallback) {
    return ''
  }
  if (!isProbablyEnglish(fallback)) {
    return fallback
  }
  switch (status) {
    case 'pass':
      return '已完成该规则检查，未发现不符合项。'
    case 'warn':
      return '该规则检查发现风险信号，建议结合原文件定位与规则溯源复核。'
    case 'fail':
      return '该规则检查发现不符合项，请结合原文件定位与规则溯源复核。'
    case 'na':
      return '当前缺少适用证据或前置条件，本规则不适用。'
    default:
      return '该规则已完成检查，请结合规则详情复核。'
  }
}

export function getRuleStatusLabel(status) {
  return STATUS_LABELS[status] ?? status ?? ''
}

export function getRuleCategoryLabel(category) {
  return CATEGORY_LABELS[category] ?? category ?? ''
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

export function getNavigationStatusLabel(status) {
  return NAVIGATION_STATUS_LABELS[status] ?? status ?? ''
}

export function getNavigationReasonLabel(reason) {
  return NAVIGATION_REASON_LABELS[reason] ?? reason ?? ''
}

export function shouldDisplayNavigationDiagnostic(status, reason, jumpPage) {
  if (typeof jumpPage !== 'number') {
    return false
  }
  if (status !== 'resolved_page' && status !== 'resolved_structural') {
    return false
  }
  if (reason === 'no_navigation_target') {
    return false
  }
  return Boolean(getNavigationStatusLabel(status) || getNavigationReasonLabel(reason))
}

export function buildRuleLocationLabelParts(document = {}) {
  const sectionSummary = (document.section_refs ?? [])
    .map((sectionRef) => [sectionRef?.outline_index, sectionRef?.section_title].filter(Boolean).join(' '))
    .filter((value) => Boolean(value))
    .join(' / ')
  return [
    document.filename,
    document.module_label ? `文档分类: ${document.module_label}` : null,
    sectionSummary,
  ].filter((value) => Boolean(value))
}

export function getVerificationStatusLabel(status) {
  return VERIFICATION_STATUS_LABELS[status] ?? status ?? ''
}

export function getCitationLabel(citation) {
  return CITATION_LABELS[citation] ?? citation ?? ''
}

function getKnownCitationLabel(citation) {
  return CITATION_LABELS[citation] ?? ''
}

const RULE_BASIS_PRESENTATION = {
  'HR-ECTD-001': {
    basisKind: 'regulation',
    basisLabel: 'eCTD 技术规范 2.3.1 序列号',
  },
  'SR-CTD-002': {
    basisKind: 'regulation',
    basisLabel: '《药品注册分类及申报资料要求》第三部分（申报资料要求）第（一）项',
  },
  'SR-REG-006': {
    basisKind: 'regulation',
    basisLabel: '《中华人民共和国药品管理法实施条例》第六条',
  },
  'HR-PARSE-001': {
    basisKind: 'system',
    basisLabel: '材料审阅契约 v1 / 文档摘要',
  },
  'HR-PARSE-002': {
    basisKind: 'system',
    basisLabel: '材料审阅契约 v1 / 解析诊断索引',
  },
  'HR-NAV-001': {
    basisKind: 'system',
    basisLabel: '材料审阅契约 v1 / 导航索引',
  },
  'HR-CTD-001': {
    basisKind: 'system',
    basisLabel: '材料审阅契约 v1 / 文档分类',
  },
  'SR-FACT-001': {
    basisKind: 'system',
    basisLabel: '材料审阅契约 v1 / 原子事实索引',
  },
  'SR-EVID-001': {
    basisKind: 'system',
    basisLabel: '材料审阅契约 v1 / 证据索引',
  },
  'SR-STRUCT-001': {
    basisKind: 'system',
    basisLabel: '材料审阅契约 v1 / 章节锚点索引',
  },
  'SR-TOC-001': {
    basisKind: 'system',
    basisLabel: '材料审阅契约 v1 / 章节树',
  },
  'SR-FACT-002': {
    basisKind: 'system',
    basisLabel: '材料审阅契约 v1 / 原子事实索引',
  },
  'SR-FACT-003': {
    basisKind: 'system',
    basisLabel: '材料审阅契约 v1 / 事实信号索引',
  },
  'SR-FACT-004': {
    basisKind: 'system',
    basisLabel: '材料审阅契约 v1 / 事实信号索引',
  },
  'SR-FACT-005': {
    basisKind: 'system',
    basisLabel: '材料审阅契约 v1 / 原子事实索引',
  },
}

export function getRuleBasisPresentation(ruleId, citation) {
  const mapped = RULE_BASIS_PRESENTATION[ruleId]
  if (mapped) {
    return mapped
  }
  return {
    basisKind: 'generic',
    basisLabel: getKnownCitationLabel(citation),
  }
}

export function buildRuleBasisDisplay(ruleId, citation, basis) {
  const fallbackPresentation = getRuleBasisPresentation(ruleId, citation)
  if (!basis) {
    if (!fallbackPresentation.basisLabel) {
      return null
    }
    return {
      labelPrefix: fallbackPresentation.basisKind === 'regulation' ? '规则溯源' : '依据说明',
      basisKind: fallbackPresentation.basisKind,
      basisLabel: fallbackPresentation.basisLabel,
      basisDetail: null,
    }
  }

  const fallbackLabel = fallbackPresentation.basisLabel
  const basisLabel = basis.basis_label?.trim?.() || fallbackLabel
  const basisDetail = basis.basis_detail?.trim?.() || null

  if (!basisLabel && !basisDetail) {
    return null
  }

  const basisKind = basis.basis_kind ?? fallbackPresentation.basisKind ?? 'generic'


  return {
    labelPrefix: basisKind === 'regulation' ? '规则溯源' : '依据说明',
    basisKind,
    basisLabel,
    basisDetail,
  }
}

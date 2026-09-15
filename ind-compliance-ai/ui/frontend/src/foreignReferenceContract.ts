const SOURCE_LABELS: Record<string, string> = {
  cn_technical_specification: '中国 eCTD 技术规范',
}

interface SourceRow {
  key: string
  label: string
  filename: string
  section: string
  heading: string
}

interface LanguageRules {
  boundary: string
  chineseIndicators: string
  chineseDescription: string
  foreignPattern: string
  foreignDescription: string
  iso639Standard: string
  iso639Count: number
}

interface StructureRules {
  colocationRequirement: string
  colocationRationale: string
  orderingRequirement: string
  orderingRationale: string
}

interface LifecycleRules {
  replaceRule: string
  rationale: string
  failureCondition: string
}

export interface ForeignReferenceContractDisplay {
  schemaVersion: string
  sourceRows: SourceRow[]
  languageRules: LanguageRules
  structureRules: StructureRules
  lifecycleRules: LifecycleRules
  deterministicChecks: string[]
  manualReviewItems: string[]
  hasManualReview: boolean
}

export function asContractRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
}

export function buildForeignReferenceContractDisplay(contract: Record<string, unknown> | null | undefined): ForeignReferenceContractDisplay | null {
  const normalized = asContractRecord(contract)
  if (!normalized) return null

  const sourceReferences = asContractRecord(normalized.source_references) ?? {}
  const languageClassification = asContractRecord(normalized.language_classification) ?? {}
  const structureRequirements = asContractRecord(normalized.structure_requirements) ?? {}
  const lifecycleConsistency = asContractRecord(normalized.lifecycle_consistency) ?? {}
  const boundary = asContractRecord(normalized.automation_boundary) ?? {}

  // 来源依据
  const sourceRows: SourceRow[] = Object.entries(sourceReferences).map(([key, value]) => {
    const source = asContractRecord(value) ?? {}
    const subsections = asContractRecord(source.subsections) ?? {}
    const subsectionKeys = Object.keys(subsections)
    const sectionsText = subsectionKeys.length > 0 ? subsectionKeys.join(', ') : String(source.section ?? '').trim()

    return {
      key,
      label: SOURCE_LABELS[key] ?? key,
      filename: String(source.source_filename ?? '').trim(),
      section: sectionsText,
      heading: String(source.heading ?? '').trim(),
    }
  })

  // 语言分类规则
  const chineseDossier = asContractRecord(languageClassification.chinese_dossier_indicators) ?? {}
  const foreignReference = asContractRecord(languageClassification.foreign_reference_indicators) ?? {}
  const iso639Requirement = asContractRecord(languageClassification.iso639_1_requirement) ?? {}

  const languageRules: LanguageRules = {
    boundary: String(languageClassification.classification_boundary ?? '').trim(),
    chineseIndicators: Array.isArray(chineseDossier.xml_lang_values)
      ? chineseDossier.xml_lang_values.map((v: unknown) => v === null ? '(缺省)' : v === '' ? '(空字符串)' : `'${v}'`).join(', ')
      : '',
    chineseDescription: String(chineseDossier.description ?? '').trim(),
    foreignPattern: String(foreignReference.xml_lang_pattern ?? '').trim(),
    foreignDescription: String(foreignReference.description ?? '').trim(),
    iso639Standard: String(iso639Requirement.standard ?? '').trim(),
    iso639Count: Array.isArray(iso639Requirement.valid_codes) ? iso639Requirement.valid_codes.length : 0,
  }

  // 结构要求
  const siblingColocation = asContractRecord(structureRequirements.sibling_colocation) ?? {}
  const orderingConstraint = asContractRecord(structureRequirements.ordering_constraint) ?? {}

  const structureRules: StructureRules = {
    colocationRequirement: String(siblingColocation.requirement ?? '').trim(),
    colocationRationale: String(siblingColocation.rationale ?? '').trim(),
    orderingRequirement: String(orderingConstraint.requirement ?? '').trim(),
    orderingRationale: String(orderingConstraint.rationale ?? '').trim(),
  }

  // 生命周期一致性
  const lifecycleRules: LifecycleRules = {
    replaceRule: String(lifecycleConsistency.replace_operation_rule ?? '').trim(),
    rationale: String(lifecycleConsistency.rationale ?? '').trim(),
    failureCondition: String(lifecycleConsistency.failure_condition ?? '').trim(),
  }

  // 自动化边界
  const deterministicChecks = Array.isArray(boundary.deterministic)
    ? boundary.deterministic.map((item: unknown) => String(item).trim()).filter(Boolean)
    : []
  const manualReviewItems = Array.isArray(boundary.manual_review)
    ? boundary.manual_review.map((item: unknown) => String(item).trim()).filter(Boolean)
    : []

  return {
    schemaVersion: String(normalized.schema_version ?? '').trim(),
    sourceRows,
    languageRules,
    structureRules,
    lifecycleRules,
    deterministicChecks,
    manualReviewItems,
    hasManualReview: manualReviewItems.length > 0,
  }
}

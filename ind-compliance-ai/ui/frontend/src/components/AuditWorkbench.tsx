import { DownloadOutlined } from '@ant-design/icons'
import { ProCard } from '@ant-design/pro-components'
import { Alert, Button, Empty, List, Select, Space, Tag, Typography } from 'antd'
import { useEffect, useMemo, useState } from 'react'
import { Document, Page } from 'react-pdf'
import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import 'katex/dist/katex.min.css'

import { buildAssetUrl } from '../api'
import {
  buildStructureAuditRecordSignature,
  buildStructureAuditPathFocusKey,
  buildStructureAuditPathFocusLabel,
  buildStructureAuditNavigationTargets,
  buildStructureAuditIssueSummary,
  buildStructureAuditGroups,
  buildStructureAuditRuleFocusList,
  getStructureAuditMissingPathRows,
  getStructureAuditPathAlignmentRows,
  getStructureAuditRootPageConflictRows,
  getStructureAuditFilenameOptions,
  getStructureAuditIssueLabel,
  getStructureAuditIssueOptions,
  resolveStructureAuditPathNavigationTarget,
  toggleStructureAuditFilterValue,
} from '../structureAudit.js'
import { resolveGuidanceRuleGroupFocusTarget } from '../scopeGuidance.js'
import { getRuleRemediationGuidanceItems } from '../ruleRemediation.js'
import {
  buildRuleHeaderPresentation,
  buildRuleBasisDisplay,
  buildRuleLocationLabelParts,
  getNavigationReasonLabel,
  getNavigationStatusLabel,
  getRuleCategoryLabel,
  getRuleStatusLabel,
  getRuleSummary,
  getVerificationStatusLabel,
  shouldDisplayNavigationDiagnostic,
} from '../rulePresentation.js'
import { buildRuleDetailDisplayRows } from '../ruleDetailPresentation.js'
import {
  DEFAULT_RULE_CHECK_STATUS_FILTER,
  filterRuleCheckItems,
  getRuleCheckFilterLabel,
  getRuleCheckStatusFilterOptions,
  type RuleCheckStatusFilter,
} from '../ruleCheckFilters.js'
import {
  buildEndToEndNavigationAuditChain,
  resolveEndToEndRuleNavigationVerification,
  resolveRuleNavigationLanding,
  resolveStructuralSelection,
} from '../ruleNavigation.js'
import { buildRegulatoryReadinessMatrixRows } from '../regulatoryReadiness.js'
import { buildSingleFileReviewSummary, getReviewPanelVisibility } from '../workbenchReviewScope.js'
import type {
  PdfAlgorithmBlock,
  BoundingBox,
  EndToEndNavigationAuditChain,
  GuidanceTargetDetail,
  PdfEquationBlock,
  PdfImageBlock,
  PdfTableAst,
  PdfTocBlock,
  PdfTocSequence,
  RuleNavigationAuditRecord,
  RuleCheckItem,
  RuleStructureAuditRecord,
  WorkbenchPayload,
} from '../types'

interface AuditWorkbenchProps {
  workbench: WorkbenchPayload | null
}

interface ActiveRuleNavigationAttempt {
  label: string
  targetPage: number | null
  targetStructuralId: string | null
  backendNavigationStatus: string | null
  backendNavigationReason: string | null
  backendAuditRecord: RuleNavigationAuditRecord | null
}

interface ActiveStructureAuditPathFocus {
  key: string
  summary: string
}

interface ActiveRuleGroupStatusFilter {
  groupId: string
  status: RuleCheckStatusFilter
}

type StructuralKind = 'table' | 'image' | 'algorithm' | 'equation' | 'toc' | 'text'

const PDF_RENDER_WIDTH = 420
const TEXT_BOUNDING_BOX_LIST_LIMIT = 60

function getSemanticRowCount(table: PdfTableAst): number | null {
  return table.row_count ?? table.data_row_count ?? null
}

function getRawRowCount(table: PdfTableAst): number | null {
  return table.raw_row_count ?? table.display_row_count ?? null
}

function getSemanticCompactionAnchors(table: PdfTableAst): string[] {
  return (table.semantic_compaction?.groups ?? [])
    .map((group) => group.anchor_text?.trim())
    .filter((anchor): anchor is string => Boolean(anchor))
}

function getBoundingBoxKind(item: BoundingBox): StructuralKind {
  if (item.semantic_role === 'algorithm_pseudocode' || item.block_type === 'algorithm' || item.id.startsWith('alg_')) {
    return 'algorithm'
  }
  if (item.semantic_role === 'display_equation' || item.block_type === 'equation') {
    return 'equation'
  }
  if (item.id.startsWith('tbl_') || item.text.startsWith('[TABLE]')) {
    return 'table'
  }
  if (
    item.id.startsWith('img_') ||
    item.id.startsWith('fig_') ||
    item.text.startsWith('[IMAGE]') ||
    item.text.startsWith('[FIGURE]')
  ) {
    return 'image'
  }
  if (item.id.startsWith('toc_') || item.text.startsWith('[TOC]')) {
    return 'toc'
  }
  return 'text'
}

function isStructuralBoundingBox(item: BoundingBox): boolean {
  return getBoundingBoxKind(item) !== 'text'
}

function getBoundingBoxKindLabel(kind: StructuralKind): string {
  switch (kind) {
    case 'algorithm':
      return '算法'
    case 'table':
      return '表格'
    case 'image':
      return '图片'
    case 'equation':
      return '公式'
    case 'toc':
      return 'TOC'
    default:
      return '文本'
  }
}

function getBoundingBoxKindTagColor(kind: StructuralKind): string {
  switch (kind) {
    case 'algorithm':
      return 'orange'
    case 'table':
      return 'purple'
    case 'image':
      return 'gold'
    case 'equation':
      return 'lime'
    case 'toc':
      return 'cyan'
    default:
      return 'default'
  }
}

function getBoundingBoxKindClassName(kind: StructuralKind): string {
  switch (kind) {
    case 'algorithm':
      return 'bbox-rect-algorithm'
    case 'table':
      return 'bbox-rect-table'
    case 'image':
      return 'bbox-rect-image'
    case 'equation':
      return 'bbox-rect-equation'
    case 'toc':
      return 'bbox-rect-toc'
    default:
      return ''
  }
}

function buildTableSummary(table: PdfTableAst): string {
  const parts: string[] = []
  const semanticRowCount = getSemanticRowCount(table)
  const rawRowCount = getRawRowCount(table)
  if (table.detection_method) {
    parts.push(table.detection_method)
  }
  if (table.col_count) {
    parts.push(`${table.col_count} 列`)
  }
  if (semanticRowCount) {
    parts.push(`语义 ${semanticRowCount} 行`)
  }
  if (rawRowCount && rawRowCount > (semanticRowCount ?? 0)) {
    parts.push(`原始 ${rawRowCount} 行`)
  }
  if (table.semantic_compaction?.applied) {
    parts.push('已做语义压缩')
  }
  return parts.join(' | ')
}

function buildTableCompactionDetail(table: PdfTableAst): string | null {
  if (!table.semantic_compaction?.applied) {
    return null
  }

  const anchors = getSemanticCompactionAnchors(table)
  const parts = [`策略: ${table.semantic_compaction.strategy}`]
  if (anchors.length > 0) {
    parts.push(`锚点: ${anchors.join(' / ')}`)
  }
  return parts.join(' | ')
}

function buildImageTitle(image: PdfImageBlock): string {
  return image.title?.trim() || image.figure_ref?.trim() || image.image_id
}

function buildImageSummary(image: PdfImageBlock): string {
  const parts: string[] = []
  if (image.figure_ref) {
    parts.push(image.figure_ref)
  }
  if (image.image_kind_guess) {
    parts.push(image.image_kind_guess)
  }
  return parts.join(' | ')
}

function buildEquationTitle(equation: PdfEquationBlock): string {
  const compact = equation.text?.trim() ?? ''
  if (!compact) {
    return equation.equation_id
  }
  if (compact.length <= 72) {
    return compact
  }
  return `${compact.slice(0, 69)}...`
}

function buildEquationSummary(equation: PdfEquationBlock): string {
  const parts: string[] = []
  if (equation.equation_label) {
    parts.push(equation.equation_label)
  }
  if (equation.source) {
    parts.push(equation.source)
  }
  return parts.join(' | ')
}

function buildAlgorithmTitle(algorithm: PdfAlgorithmBlock): string {
  return algorithm.title?.trim() || algorithm.algorithm_ref?.trim() || algorithm.algorithm_id
}

function buildAlgorithmSummary(algorithm: PdfAlgorithmBlock): string {
  const parts: string[] = []
  if (algorithm.algorithm_ref) {
    parts.push(algorithm.algorithm_ref)
  }
  if (algorithm.line_count && algorithm.line_count > 0) {
    parts.push(`${algorithm.line_count} 行`)
  }
  if (algorithm.continued_from_previous_page) {
    parts.push('上页续接')
  }
  if (algorithm.continues_to_next_page) {
    parts.push('续至下页')
  }
  return parts.join(' | ')
}

function buildTocTitle(toc: PdfTocBlock): string {
  return toc.title?.trim() || toc.toc_id
}

function buildTocSummary(toc: PdfTocBlock): string {
  if (toc.entry_count && toc.entry_count > 0) {
    return `${toc.entry_count} 条目录项`
  }
  return ''
}

function buildTocSequenceTitle(sequence: PdfTocSequence): string {
  return sequence.title?.trim() || sequence.toc_sequence_id
}

function buildTocSequencePageLabel(sequence: PdfTocSequence): string | null {
  const pages = sequence.pages ?? []
  if (pages.length === 0) {
    return null
  }
  if (pages.length === 1) {
    return `page ${pages[0]}`
  }
  return `pages ${pages[0]}-${pages[pages.length - 1]}`
}

function buildTocSequenceSummary(sequence: PdfTocSequence, currentPage: number): string {
  const parts: string[] = []
  const pageLabel = buildTocSequencePageLabel(sequence)
  if (pageLabel) {
    parts.push(pageLabel)
  }
  if (sequence.entry_count && sequence.entry_count > 0) {
    parts.push(`${sequence.entry_count} items`)
  }
  if (sequence.root_entry_count && sequence.root_entry_count > 0) {
    parts.push(`${sequence.root_entry_count} roots`)
  }
  if (sequence.max_branching_factor && sequence.max_branching_factor > 0) {
    parts.push(`max branch ${sequence.max_branching_factor}`)
  }
  if ((sequence.pages ?? []).includes(currentPage)) {
    const pageIndex = (sequence.pages ?? []).indexOf(currentPage)
    parts.push(`current page ${pageIndex + 1}/${(sequence.pages ?? []).length}`)
  }
  return parts.join(' | ')
}

function buildTocSequenceDetail(sequence: PdfTocSequence): string | null {
  const rootSections = (sequence.navigation_summary?.root_sections ?? [])
    .slice(0, 4)
    .map((section) => {
      const outline = section.outline_index?.trim()
      const text = section.text?.trim()
      if (outline && text) {
        return `${outline} ${text}`
      }
      return outline || text || ''
    })
    .filter((value): value is string => Boolean(value))

  const parts: string[] = []
  if (rootSections.length > 0) {
    parts.push(`roots: ${rootSections.join(' / ')}`)
  }
  if (
    sequence.navigation_summary?.cross_page_parent_link_count &&
    sequence.navigation_summary.cross_page_parent_link_count > 0
  ) {
    parts.push(
      `cross-page parent links: ${sequence.navigation_summary.cross_page_parent_link_count}`,
    )
  }
  return parts.length > 0 ? parts.join(' | ') : null
}

function buildTocBlockSequenceHint(
  toc: PdfTocBlock,
  tocSequenceLookup: Map<string, PdfTocSequence>,
): string | null {
  if (!toc.toc_sequence_id) {
    return null
  }
  const sequence = tocSequenceLookup.get(toc.toc_sequence_id)
  if (!sequence) {
    return `sequence: ${toc.toc_sequence_id}`
  }

  const parts = [`sequence: ${buildTocSequenceTitle(sequence)}`]
  const pageLabel = buildTocSequencePageLabel(sequence)
  if (pageLabel) {
    parts.push(pageLabel)
  }
  return parts.join(' | ')
}

function getRuleStatusColor(status: string): string {
  switch (status) {
    case 'pass':
      return 'green'
    case 'warn':
      return 'orange'
    case 'fail':
      return 'red'
    case 'na':
      return 'default'
    default:
      return 'blue'
  }
}

function getReviewProjectionStatusLabel(status?: string | null): string {
  switch (status) {
    case 'ready':
      return '已就绪'
    case 'ready_with_review_items':
      return '有待复核项'
    case 'asset_missing':
      return '缺少演示材料'
    case 'ready_for_demo':
      return '可演示'
    case 'review_required':
    case 'human_review':
      return '建议复核'
    case 'prerequisite_prompt':
    case 'prerequisite_required':
      return '需补充前置条件'
    case 'not_available':
      return '暂不可用'
    case 'consistent':
      return '一致'
    case 'present':
      return '已提供'
    case 'missing':
      return '待补充'
    case 'local_evidence_present':
      return '已有本地证据'
    case 'missing_required_prerequisite':
      return '缺少前置条件'
    case 'high':
      return '高可信'
    case 'medium':
      return '中等可信'
    case 'none':
      return '无本地证据'
    case 'demo_ready_in_progress':
      return '演示能力建设中'
    case 'phase_a_demo_workbench':
      return 'A 阶段演示工作台'
    case 'controlled_customer_demo':
      return '受控客户演示'
    case 'ectd_sequence_batch':
      return 'eCTD 序列批量资料'
    case 'customer_readiness_walkthrough':
      return '客户演示路径'
    case 'single_sequence':
      return '单序列资料'
    case 'sequence_batch':
      return '序列批量资料'
    default:
      return status || '未判定'
  }
}

function getRuleCategoryColor(category: string): string {
  return category === 'hard' ? 'volcano' : 'geekblue'
}

function getGuidancePriorityColor(priority: string): string {
  switch (priority) {
    case 'priority':
      return 'volcano'
    case 'attention':
      return 'gold'
    default:
      return 'blue'
  }
}

function getGuidancePriorityLabel(priority: string): string {
  switch (priority) {
    case 'priority':
      return '优先处理'
    case 'attention':
      return '重点关注'
    default:
      return '整改建议'
  }
}

function getGuidanceTargetTypeLabel(targetType: string): string {
  switch (targetType) {
    case 'file':
      return '检查文件'
    case 'metadata_field_group':
      return '核对字段'
    case 'metadata_field':
      return '检查字段'
    case 'folder':
      return '目录结构'
    case 'upload_requirement':
      return '上传要求'
    case 'rule_scope':
      return '关注作用域'
    default:
      return '目标对象'
  }
}

function getRemediationTargetTypeLabel(targetType: string): string {
  switch (targetType) {
    case 'node_extension_title':
      return '扩展节点标题'
    case 'allowed_title':
      return '允许标题'
    case 'leaf_href':
      return 'Leaf 路径'
    case 'path_pattern':
      return '建议路径模式'
    case 'xml_parent':
      return '目标父节点'
    case 'document_file':
      return '定位文件'
    default:
      return getGuidanceTargetTypeLabel(targetType)
  }
}

function buildRemediationTargetContext(detail: GuidanceTargetDetail): string | null {
  const contextParts = [
    detail.source_extension_title ? `扩展标题 ${detail.source_extension_title}` : null,
    detail.source_leaf_titles && detail.source_leaf_titles.length > 0
      ? `leaf 标题 ${detail.source_leaf_titles.join(' / ')}`
      : null,
    detail.source_parent_pointer ? `XML 位置 ${detail.source_parent_pointer}` : null,
  ].filter((value): value is string => Boolean(value))

  return contextParts.length > 0 ? contextParts.join(' | ') : null
}

function getExtensionIssueLabel(issueCode: string): string {
  switch (issueCode) {
    case 'invalid_extension_title':
      return '标题不合法'
    case 'invalid_leaf_href':
      return '路径不合法'
    case 'misplaced_extension':
      return '挂载位置错误'
    default:
      return issueCode || '未知问题'
  }
}

function formatRatio(value?: number | null): string | null {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return null
  }
  return `${(value * 100).toFixed(0)}%`
}

function formatStructureAuditPathRow(row: {
  outlinePath: string | null
  textPath: string | null
  pageLocatorValue: number | null
  navigationPage: number | null
  level: number | null
  nearestBodyAnchorOutlineIndex?: string | null
}): string {
  const primaryPath = row.textPath || row.outlinePath || '未解析路径'
  const detailParts = [
    row.textPath && row.outlinePath ? `编号 ${row.outlinePath}` : null,
    typeof row.pageLocatorValue === 'number' ? `目录标注页 ${row.pageLocatorValue}` : null,
    typeof row.navigationPage === 'number' ? `TOC页 ${row.navigationPage}` : null,
    typeof row.level === 'number' ? `层级 ${row.level}` : null,
    row.nearestBodyAnchorOutlineIndex ? `最近正文锚点 ${row.nearestBodyAnchorOutlineIndex}` : null,
  ].filter((value): value is string => Boolean(value))
  return detailParts.length > 0 ? `${primaryPath}（${detailParts.join('；')}）` : primaryPath
}

function getStructureAuditAlignmentStatusLabel(status?: string | null): string {
  switch (status) {
    case 'matched':
      return '已匹配'
    case 'nearest_parent':
      return '需核对'
    case 'missing':
      return '未定位'
    default:
      return status || '-'
  }
}

function findNavigationAuditRecord(
  auditRecords: RuleNavigationAuditRecord[],
  ruleId: string,
  targetKind: 'matched_document' | 'weak_signal_snippet',
  filename?: string,
  targetPage?: number,
  targetStructuralId?: string,
  evidenceId?: string,
): RuleNavigationAuditRecord | null {
  return (
    auditRecords.find((record) => {
      if (record.rule_id !== ruleId) {
        return false
      }
      if (record.target_kind !== targetKind) {
        return false
      }
      if (filename && record.filename !== filename) {
        return false
      }
      if (typeof targetPage === 'number' && (record.target_page ?? null) !== targetPage) {
        return false
      }
      if (targetStructuralId && (record.target_structural_id ?? null) !== targetStructuralId) {
        return false
      }
      if (evidenceId && (record.evidence_id ?? null) !== evidenceId) {
        return false
      }
      return true
    }) ?? null
  )
}

function renderRuleDetails(
  rule: RuleCheckItem,
  navigationAuditRecords: RuleNavigationAuditRecord[],
  onNavigate: (
    page?: number,
    structuralId?: string,
    label?: string,
    backendNavigationStatus?: string,
    backendNavigationReason?: string,
    backendAuditRecord?: RuleNavigationAuditRecord | null,
  ) => void,
) {
  const details = rule.details ?? {}

  const ruleDetailDisplayRows = buildRuleDetailDisplayRows(details)
  const remediationGuidanceItems = getRuleRemediationGuidanceItems(details.remediation_guidance)
  const basisDisplay = buildRuleBasisDisplay(
    rule.rule_id,
    rule.citation ?? details.citation_anchor ?? '',
    rule.basis
      ? {
          basis_kind: rule.basis.basis_kind,
          basis_label: rule.basis.basis_label,
          basis_detail: rule.basis.basis_detail,
        }
      : null,
  )

  return (
    <Space direction="vertical" size={4} style={{ width: '100%' }}>
      {ruleDetailDisplayRows.map((row) =>
        row.kind === 'tag' ? (
          <Space key={`${row.label}-${row.value}`} size={6} wrap>
            <Typography.Text className="rule-detail-line">{row.label}:</Typography.Text>
            <Tag color={row.color ?? 'default'}>{row.value}</Tag>
          </Space>
        ) : (
          <Typography.Text key={`${row.label}-${row.value}`} type="secondary" className="rule-detail-line">
            {row.label}: {row.value}
          </Typography.Text>
        ),
      )}
      {basisDisplay ? (
        <div className="rule-detail-block">
          <Typography.Text strong>{basisDisplay.labelPrefix}</Typography.Text>
          <Typography.Text type="secondary" className="rule-detail-line">
            {basisDisplay.basisLabel}
          </Typography.Text>
          {basisDisplay.basisDetail ? (
            <Typography.Text className="rule-detail-line">{basisDisplay.basisDetail}</Typography.Text>
          ) : null}
        </div>
      ) : null}
      {(details.matched_documents ?? []).length > 0 ? (
        <div className="rule-detail-block">
          <Typography.Text strong>原文件定位</Typography.Text>
          <div className="rule-detail-chip-list">
            {(details.matched_documents ?? []).map((document, index) => {
              const labelParts = buildRuleLocationLabelParts(document)
              return (
                <Tag key={`${document.document_id ?? document.filename ?? 'doc'}-${index}`} color="blue">
                  {labelParts.join(' | ')}
                </Tag>
              )
            })}
          </div>
          <Space size={[8, 8]} wrap>
            {(details.matched_documents ?? [])
              .filter((document) => typeof document.jump_page === 'number')
              .map((document, index) => {
                const auditRecord = findNavigationAuditRecord(
                  navigationAuditRecords,
                  rule.rule_id,
                  'matched_document',
                  document.filename,
                  document.jump_page,
                  document.structural_id,
                )
                return (
                  <Button
                    key={`${document.document_id ?? document.filename ?? 'doc'}-jump-${index}`}
                    size="small"
                    type="link"
                    onClick={() =>
                      onNavigate(
                        document.jump_page,
                        document.structural_id,
                        `rule:${rule.rule_id}:${document.filename ?? 'document'}`,
                        document.navigation_status,
                        document.navigation_reason,
                        auditRecord,
                      )
                    }
                  >
                    跳转到原文件页 {document.filename ?? 'document'} p.{document.jump_page}
                  </Button>
                )
              })}
          </Space>
          <Space direction="vertical" size={2} style={{ width: '100%' }}>
            {(details.matched_documents ?? []).map((document, index) =>
              shouldDisplayNavigationDiagnostic(
                document.navigation_status,
                document.navigation_reason,
                document.jump_page,
              ) ? (
                <Typography.Text
                  key={`${document.document_id ?? document.filename ?? 'doc'}-nav-${index}`}
                  type="secondary"
                  className="rule-detail-line"
                >
                  导航状态: {getNavigationStatusLabel(document.navigation_status ?? '')} |{' '}
                  {getNavigationReasonLabel(document.navigation_reason ?? '')}
                </Typography.Text>
              ) : null,
            )}
          </Space>
        </div>
      ) : null}
      {(details.weak_signal_snippets ?? []).length > 0 ? (
        <div className="rule-detail-block">
          <Typography.Text strong>问题内容片段</Typography.Text>
          <Space direction="vertical" size={4} style={{ width: '100%' }}>
            {(details.weak_signal_snippets ?? []).map((snippet, index) => {
              const jumpPage = snippet.jump_page ?? snippet.page
              const auditRecord = findNavigationAuditRecord(
                navigationAuditRecords,
                rule.rule_id,
                'weak_signal_snippet',
                snippet.filename,
                jumpPage,
                snippet.structural_id,
                snippet.evidence_id,
              )
              return (
                <Space
                  key={`${snippet.evidence_id ?? snippet.filename ?? 'snippet'}-${index}`}
                  direction="vertical"
                  size={2}
                  style={{ width: '100%' }}
                >
                  <Typography.Text className="rule-detail-line">
                    {(snippet.filename ?? 'unknown-document') + (jumpPage ? ` p.${jumpPage}` : '')}: {snippet.preview}
                  </Typography.Text>
                  {typeof jumpPage === 'number' ? (
                    <Button
                      size="small"
                      type="link"
                      onClick={() =>
                        onNavigate(
                          jumpPage,
                          snippet.structural_id,
                          `rule:${rule.rule_id}:${snippet.filename ?? 'snippet'}`,
                          snippet.navigation_status,
                          snippet.navigation_reason,
                          auditRecord,
                        )
                      }
                    >
                      跳转到片段页
                    </Button>
                  ) : null}
                  {shouldDisplayNavigationDiagnostic(snippet.navigation_status, snippet.navigation_reason, jumpPage) ? (
                    <Typography.Text type="secondary" className="rule-detail-line">
                      导航状态: {getNavigationStatusLabel(snippet.navigation_status ?? '')} |{' '}
                      {getNavigationReasonLabel(snippet.navigation_reason ?? '')}
                    </Typography.Text>
                  ) : null}
                </Space>
              )
            })}
          </Space>
        </div>
      ) : null}
      {(details.matched_documents ?? []).some(
        (document) => (document.non_embedded_non_standard_font_records ?? []).length > 0,
      ) ? (
        <div className="rule-detail-block">
          <Typography.Text strong>字体证据</Typography.Text>
          <Space direction="vertical" size={6} style={{ width: '100%' }}>
            {(details.matched_documents ?? []).flatMap((document, documentIndex) =>
              (document.non_embedded_non_standard_font_records ?? []).map((fontRecord, fontIndex) => {
                const pageNumbers = (fontRecord.page_numbers ?? []).filter(
                  (pageNumber): pageNumber is number => typeof pageNumber === 'number' && pageNumber > 0,
                )
                const firstPage = pageNumbers[0]
                return (
                  <Space
                    key={`${document.document_id ?? document.filename ?? 'doc'}-font-${fontRecord.font_label ?? fontIndex}-${documentIndex}`}
                    direction="vertical"
                    size={3}
                    style={{ width: '100%' }}
                  >
                    <Typography.Text className="rule-detail-line">
                      {document.filename ?? 'unknown-document'}: {fontRecord.font_label ?? '未知字体'}
                    </Typography.Text>
                    <Space wrap>
                      <Tag color="orange">非标准字体</Tag>
                      <Tag color={fontRecord.is_embedded ? 'green' : 'red'}>
                        {fontRecord.is_embedded ? '已嵌入' : '未嵌入'}
                      </Tag>
                      {fontRecord.font_type ? <Tag>{fontRecord.font_type}</Tag> : null}
                      {fontRecord.resource_name ? <Tag>资源名 {fontRecord.resource_name}</Tag> : null}
                      {fontRecord.xref ? <Tag>xref {fontRecord.xref}</Tag> : null}
                      {pageNumbers.length > 0 ? <Tag>出现页 {pageNumbers.join(' / ')}</Tag> : null}
                    </Space>
                    <Typography.Text type="secondary" className="rule-detail-line">
                      规则关注的是 PDF 字体资源是否为非标准字体且未嵌入；请结合字体名、嵌入状态和出现页复核是否为可接受字体。
                    </Typography.Text>
                    {typeof firstPage === 'number' ? (
                      <Button
                        size="small"
                        type="link"
                        onClick={() =>
                          onNavigate(
                            firstPage,
                            document.structural_id,
                            `rule:${rule.rule_id}:${document.filename ?? 'font'}:${fontRecord.font_label ?? fontIndex}`,
                            document.navigation_status,
                            document.navigation_reason,
                            null,
                          )
                        }
                        style={{ paddingInline: 0, width: 'fit-content' }}
                      >
                        跳转到字体出现页 p.{firstPage}
                      </Button>
                    ) : null}
                  </Space>
                )
              }),
            )}
          </Space>
        </div>
      ) : null}
      {details.llm_decision ? (
        <Typography.Text type="secondary" className="rule-detail-line">
          LLM: {details.llm_decision.provider ?? 'unknown'} | {details.llm_decision.decision ?? 'unknown'} |{' '}
          {details.llm_decision.reason ?? 'no reason'}
        </Typography.Text>
      ) : null}
      {(details.extension_issue_bundles ?? []).length > 0 ? (
        <div className="rule-detail-block">
          <Typography.Text strong>节点整改定位</Typography.Text>
          <Space direction="vertical" size={6} style={{ width: '100%' }}>
            {(details.extension_issue_bundles ?? []).map((bundle) => (
              <div key={bundle.bundle_id} className="rule-detail-block">
                <Space direction="vertical" size={3} style={{ width: '100%' }}>
                  <Typography.Text className="rule-detail-line">
                    [{bundle.source_filename ?? 'unknown-document'}] {bundle.extension_title ?? '(missing title)'}
                  </Typography.Text>
                  <div className="rule-detail-chip-list">
                    {(bundle.issue_codes ?? []).map((issueCode) => (
                      <Tag key={`${bundle.bundle_id}-${issueCode}`} color="orange">
                        {getExtensionIssueLabel(issueCode)}
                      </Tag>
                    ))}
                    {bundle.primary_issue_label ? <Tag color="red">先处理: {bundle.primary_issue_label}</Tag> : null}
                  </div>
                  {(bundle.recommended_fix_labels ?? []).length > 0 ? (
                    <Typography.Text type="secondary" className="rule-detail-line">
                      建议顺序: {(bundle.recommended_fix_labels ?? []).join(' -> ')}
                    </Typography.Text>
                  ) : null}
                  {(bundle.issue_diff_rows ?? []).length > 0 ? (
                    <Space direction="vertical" size={2} style={{ width: '100%' }}>
                      {(bundle.issue_diff_rows ?? []).map((row, index) => (
                        <div key={`${bundle.bundle_id}-diff-${row.issue_code}-${index}`}>
                          <Typography.Text type="secondary" className="rule-detail-line">
                            {row.issue_label}
                            {row.field_path ? ` | 字段 ${row.field_path}` : ''}: 当前 {row.current_value ?? '(unknown)'} | 目标{' '}
                            {row.target_value ?? '(review required)'}
                            {row.note ? ` | ${row.note}` : ''}
                          </Typography.Text>
                          {row.recommended_target_value ? (
                            <Typography.Text type="secondary" className="rule-detail-line">
                              推荐目标: {row.recommended_target_value}
                            </Typography.Text>
                          ) : null}
                          {(row.target_value_candidates ?? []).length > 0 ? (
                            <Typography.Text type="secondary" className="rule-detail-line">
                              候选目标: {(row.target_value_candidates ?? []).join(' / ')}
                            </Typography.Text>
                          ) : null}
                          {row.recommendation_basis ? (
                            <Typography.Text type="secondary" className="rule-detail-line">
                              推荐依据: {row.recommendation_basis}
                            </Typography.Text>
                          ) : null}
                          {row.has_recommendation_conflict && row.recommendation_conflict_summary ? (
                            <Typography.Text type="warning" className="rule-detail-line">
                              证据冲突: {row.recommendation_conflict_summary}
                            </Typography.Text>
                          ) : null}
                          {row.tie_break_guidance_title ? (
                            <div style={{ marginTop: 4 }}>
                              <Typography.Text strong>{row.tie_break_guidance_title}</Typography.Text>
                              {(row.tie_break_focus_items ?? []).length > 0 ? (
                                <div className="rule-detail-chip-list" style={{ marginTop: 4 }}>
                                  {(row.tie_break_focus_items ?? []).map((item, itemIndex) => (
                                    <Tag key={`${bundle.bundle_id}-diff-focus-${row.issue_code}-${index}-${itemIndex}`}>
                                      {item.label}: {item.value}
                                    </Tag>
                                  ))}
                                </div>
                              ) : null}
                              {(row.tie_break_guidance_steps ?? []).length > 0 ? (
                                <ol style={{ margin: '4px 0 0', paddingLeft: 20 }}>
                                  {(row.tie_break_guidance_steps ?? []).map((step, stepIndex) => (
                                    <li key={`${bundle.bundle_id}-diff-tiebreak-${row.issue_code}-${index}-${stepIndex}`}>
                                      <Typography.Text type="secondary">{step}</Typography.Text>
                                    </li>
                                  ))}
                                </ol>
                              ) : null}
                            </div>
                          ) : null}
                          {(row.tie_break_candidate_comparisons ?? []).length > 0 ? (
                            <div style={{ marginTop: 4 }}>
                              {(row.tie_break_candidate_comparisons ?? []).map((comparison, comparisonIndex) => (
                                <div key={`${bundle.bundle_id}-diff-compare-${row.issue_code}-${index}-${comparisonIndex}`}>
                                  <Typography.Text strong>{comparison.candidate_title}</Typography.Text>
                                  {(comparison.supports ?? []).length > 0 ? (
                                    <ul style={{ margin: '4px 0 0', paddingLeft: 20 }}>
                                      {(comparison.supports ?? []).map((item, itemIndex) => (
                                        <li key={`${bundle.bundle_id}-diff-compare-support-${comparisonIndex}-${itemIndex}`}>
                                          <Typography.Text type="secondary">支持: {item}</Typography.Text>
                                        </li>
                                      ))}
                                    </ul>
                                  ) : null}
                                  {(comparison.concerns ?? []).length > 0 ? (
                                    <ul style={{ margin: '4px 0 0', paddingLeft: 20 }}>
                                      {(comparison.concerns ?? []).map((item, itemIndex) => (
                                        <li key={`${bundle.bundle_id}-diff-compare-concern-${comparisonIndex}-${itemIndex}`}>
                                          <Typography.Text type="secondary">反证: {item}</Typography.Text>
                                        </li>
                                      ))}
                                    </ul>
                                  ) : null}
                                </div>
                              ))}
                            </div>
                          ) : null}
                          {row.suggested_action_title ? (
                            <div style={{ marginTop: 4 }}>
                              <Typography.Text strong>{row.suggested_action_title}</Typography.Text>
                              {(row.suggested_action_steps ?? []).length > 0 ? (
                                <ol style={{ margin: '4px 0 0', paddingLeft: 20 }}>
                                  {(row.suggested_action_steps ?? []).map((step, stepIndex) => (
                                    <li key={`${bundle.bundle_id}-diff-step-${row.issue_code}-${index}-${stepIndex}`}>
                                      <Typography.Text type="secondary">{step}</Typography.Text>
                                    </li>
                                  ))}
                                </ol>
                              ) : null}
                            </div>
                          ) : null}
                          {(row.verification_checks ?? []).length > 0 ? (
                            <div style={{ marginTop: 4 }}>
                              <Typography.Text strong>修后核对</Typography.Text>
                              <ul style={{ margin: '4px 0 0', paddingLeft: 20 }}>
                                {(row.verification_checks ?? []).map((check, checkIndex) => (
                                  <li key={`${bundle.bundle_id}-diff-check-${row.issue_code}-${index}-${checkIndex}`}>
                                    <Typography.Text type="secondary">{check}</Typography.Text>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          ) : null}
                          {row.suggested_snippet ? (
                            <pre
                              style={{
                                margin: '4px 0 0',
                                padding: '8px 10px',
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word',
                                borderRadius: 6,
                                background: '#f5f5f5',
                                color: 'rgba(0, 0, 0, 0.88)',
                              }}
                            >
                              {row.suggested_snippet}
                            </pre>
                          ) : null}
                        </div>
                      ))}
                    </Space>
                  ) : null}
                  {(bundle.leaf_titles ?? []).length > 0 ? (
                    <Typography.Text type="secondary" className="rule-detail-line">
                      leaf 标题: {(bundle.leaf_titles ?? []).join(' / ')}
                    </Typography.Text>
                  ) : null}
                  {(bundle.invalid_leaf_hrefs ?? []).length > 0 ? (
                    <Typography.Text type="secondary" className="rule-detail-line">
                      无效 href: {(bundle.invalid_leaf_hrefs ?? []).join(' / ')}
                    </Typography.Text>
                  ) : null}
                  {bundle.parent_pointer ? (
                    <Typography.Text type="secondary" className="rule-detail-line">
                      XML 位置: {bundle.parent_pointer}
                    </Typography.Text>
                  ) : null}
                </Space>
              </div>
            ))}
          </Space>
        </div>
      ) : null}
      {(details.matched_extension_titles ?? []).length > 0 ||
      (details.invalid_extension_titles ?? []).length > 0 ||
      (details.invalid_leaf_hrefs ?? []).length > 0 ||
      (details.misplaced_extension_titles ?? []).length > 0 ? (
        <div className="rule-detail-block">
          <Typography.Text strong>3.2.R 扩展节点诊断</Typography.Text>
          <Space direction="vertical" size={4} style={{ width: '100%' }}>
            {(details.matched_extension_titles ?? []).length > 0 ? (
              <Space direction="vertical" size={2} style={{ width: '100%' }}>
                <Typography.Text type="secondary" className="rule-detail-line">
                  已识别扩展节点标题
                </Typography.Text>
                <div className="rule-detail-chip-list">
                  {(details.matched_extension_titles ?? []).map((title) => (
                    <Tag key={`matched-extension-${title}`} color="blue">
                      {title}
                    </Tag>
                  ))}
                </div>
              </Space>
            ) : null}
            {(details.invalid_extension_titles ?? []).length > 0 ? (
              <Space direction="vertical" size={2} style={{ width: '100%' }}>
                <Typography.Text type="secondary" className="rule-detail-line">
                  标题不符合允许集合
                </Typography.Text>
                <div className="rule-detail-chip-list">
                  {(details.invalid_extension_titles ?? []).map((title) => (
                    <Tag key={`invalid-extension-${title}`} color="orange">
                      {title}
                    </Tag>
                  ))}
                </div>
              </Space>
            ) : null}
            {(details.invalid_leaf_hrefs ?? []).length > 0 ? (
              <Space direction="vertical" size={2} style={{ width: '100%' }}>
                <Typography.Text type="secondary" className="rule-detail-line">
                  Leaf 路径与 3.2.R 区域语义不一致
                </Typography.Text>
                <div className="rule-detail-chip-list">
                  {(details.invalid_leaf_hrefs ?? []).map((href) => (
                    <Tag key={`invalid-leaf-href-${href}`} color="volcano">
                      {href}
                    </Tag>
                  ))}
                </div>
              </Space>
            ) : null}
            {(details.misplaced_extension_titles ?? []).length > 0 ? (
              <Space direction="vertical" size={2} style={{ width: '100%' }}>
                <Typography.Text type="secondary" className="rule-detail-line">
                  扩展节点未位于 3.2.R 区域结构下
                </Typography.Text>
                <div className="rule-detail-chip-list">
                  {(details.misplaced_extension_titles ?? []).map((title) => (
                    <Tag key={`misplaced-extension-${title}`} color="red">
                      {title}
                    </Tag>
                  ))}
                </div>
              </Space>
            ) : null}
            {(details.allowed_extension_titles ?? []).length > 0 ? (
              <Space direction="vertical" size={2} style={{ width: '100%' }}>
                <Typography.Text type="secondary" className="rule-detail-line">
                  允许标题集合
                </Typography.Text>
                <div className="rule-detail-chip-list">
                  {(details.allowed_extension_titles ?? []).map((title) => (
                    <Tag key={`allowed-extension-${title}`} color="green">
                      {title}
                    </Tag>
                  ))}
                </div>
              </Space>
            ) : null}
            {details.expected_leaf_href_pattern || details.expected_leaf_href_note ? (
              <Space direction="vertical" size={2} style={{ width: '100%' }}>
                <Typography.Text type="secondary" className="rule-detail-line">
                  建议路径语义
                </Typography.Text>
                {details.expected_leaf_href_pattern ? (
                  <Tag color="geekblue">{details.expected_leaf_href_pattern}</Tag>
                ) : null}
                {details.expected_leaf_href_note ? (
                  <Typography.Text type="secondary" className="rule-detail-line">
                    {details.expected_leaf_href_note}
                  </Typography.Text>
                ) : null}
              </Space>
            ) : null}
          </Space>
        </div>
      ) : null}
      {remediationGuidanceItems.length > 0 ? (
        <div className="rule-detail-block">
          <Typography.Text strong>整改指引</Typography.Text>
          <Space direction="vertical" size={8} style={{ width: '100%' }}>
            {remediationGuidanceItems.map((guidance, guidanceIndex) => {
              const guidanceTargetDetails = guidance.guidance_target_details ?? []
              const groupedTargetDetails = guidanceTargetDetails.reduce<
                Array<{ targetType: string; items: typeof guidanceTargetDetails }>
              >((groups, detail) => {
                const existingGroup = groups.find((group) => group.targetType === detail.target_type)
                if (existingGroup) {
                  existingGroup.items.push(detail)
                  return groups
                }
                groups.push({ targetType: detail.target_type, items: [detail] })
                return groups
              }, [])

              return (
                <div
                  key={`remediation-guidance-${guidance.guidance_code || guidanceIndex}`}
                  className="rule-detail-block"
                >
                  <Space direction="vertical" size={4} style={{ width: '100%' }}>
                    <Space wrap>
                      <Typography.Text strong>{guidance.guidance_title || '整改建议'}</Typography.Text>
                      <Tag color={getGuidancePriorityColor(guidance.guidance_priority)}>
                        {getGuidancePriorityLabel(guidance.guidance_priority)}
                      </Tag>
                    </Space>
                    {guidance.guidance_summary ? (
                      <Typography.Text type="secondary" className="rule-detail-line">
                        {guidance.guidance_summary}
                      </Typography.Text>
                    ) : null}
                    {groupedTargetDetails.length > 0 ? (
                      <Space direction="vertical" size={4} style={{ width: '100%' }}>
                        {groupedTargetDetails.map((group) => (
                          <div key={`remediation-group-${guidance.guidance_code}-${group.targetType}`}>
                            <Typography.Text type="secondary" className="rule-detail-line">
                              {getRemediationTargetTypeLabel(group.targetType)}
                            </Typography.Text>
                            <div className="rule-detail-chip-list">
                              {group.items.map((detail) => (
                                <Tag
                                  key={`remediation-detail-${guidance.guidance_code}-${group.targetType}-${detail.label}`}
                                  color={group.targetType === 'allowed_title' ? 'green' : 'geekblue'}
                                >
                                  {detail.label}
                                </Tag>
                              ))}
                            </div>
                            <Space direction="vertical" size={2} style={{ width: '100%' }}>
                              {group.items.map((detail) => (
                                <Space
                                  key={`remediation-detail-description-${guidance.guidance_code}-${group.targetType}-${detail.label}`}
                                  direction="vertical"
                                  size={0}
                                  style={{ width: '100%' }}
                                >
                                  <Typography.Text type="secondary" className="rule-detail-line">
                                    {detail.source_filename ? `[${detail.source_filename}] ` : ''}
                                    {detail.label}: {detail.description}
                                  </Typography.Text>
                                  {buildRemediationTargetContext(detail) ? (
                                    <Typography.Text type="secondary" className="rule-detail-line">
                                      {buildRemediationTargetContext(detail)}
                                    </Typography.Text>
                                  ) : null}
                                </Space>
                              ))}
                            </Space>
                          </div>
                        ))}
                      </Space>
                    ) : (guidance.guidance_targets ?? []).length > 0 ? (
                      <div className="rule-detail-chip-list">
                        {(guidance.guidance_targets ?? []).map((target) => (
                          <Tag key={`remediation-target-${guidance.guidance_code}-${target}`} color="geekblue">
                            {target}
                          </Tag>
                        ))}
                      </div>
                    ) : null}
                    <Space direction="vertical" size={2} style={{ width: '100%' }}>
                      {(guidance.guidance_steps ?? []).map((step, stepIndex) => (
                        <Typography.Text
                          key={`remediation-step-${guidance.guidance_code}-${stepIndex + 1}`}
                          className="rule-detail-line"
                        >
                          {stepIndex + 1}. {step}
                        </Typography.Text>
                      ))}
                    </Space>
                  </Space>
                </div>
              )
            })}
          </Space>
        </div>
      ) : null}
      {(details.structure_audit_rows ?? []).length > 0 ? (
        <div className="rule-detail-block">
          <Typography.Text strong>结构审计</Typography.Text>
          <Space direction="vertical" size={4} style={{ width: '100%' }}>
            {(details.structure_audit_rows ?? []).map((row, index) => (
              <Space
                key={`${row.document_id ?? row.filename ?? 'structure'}-${index}`}
                direction="vertical"
                size={2}
                style={{ width: '100%' }}
              >
                <Typography.Text className="rule-detail-line">
                  {row.filename ?? 'unknown-document'}
                </Typography.Text>
                <Space size={[8, 8]} wrap>
                  <Tag color={row.alignment_ready ? 'green' : 'orange'}>
                    {row.alignment_ready ? '结构通过' : '结构预警'}
                  </Tag>
                  {formatRatio(row.root_outline_coverage_ratio) ? (
                    <Tag>根章节覆盖 {formatRatio(row.root_outline_coverage_ratio)}</Tag>
                  ) : null}
                  {formatRatio(row.direct_child_coverage_ratio) ? (
                    <Tag>直接子级覆盖 {formatRatio(row.direct_child_coverage_ratio)}</Tag>
                  ) : null}
                  {formatRatio(row.bounded_subtree_coverage_ratio) ? (
                    <Tag>子树覆盖 {formatRatio(row.bounded_subtree_coverage_ratio)}</Tag>
                  ) : null}
                  {row.root_page_order_ready === false ? <Tag color="orange">根章节页序冲突</Tag> : null}
                  {row.root_page_offset_ready === false ? <Tag color="orange">根章节偏移冲突</Tag> : null}
                  {row.root_page_span_ready === false ? <Tag color="orange">根章节区间冲突</Tag> : null}
                </Space>
                {(row.missing_body_root_outline_indices ?? []).length > 0 ? (
                  <Typography.Text type="secondary" className="rule-detail-line">
                    缺失根章节: {(row.missing_body_root_outline_indices ?? []).join(' / ')}
                  </Typography.Text>
                ) : null}
                {(row.missing_body_direct_child_outline_indices ?? []).length > 0 ? (
                  <Typography.Text type="secondary" className="rule-detail-line">
                    缺失直接子级: {(row.missing_body_direct_child_outline_indices ?? []).join(' / ')}
                  </Typography.Text>
                ) : null}
                {(row.missing_body_bounded_subtree_outline_indices ?? []).length > 0 ? (
                  <Typography.Text type="secondary" className="rule-detail-line">
                    缺失子树节点: {(row.missing_body_bounded_subtree_outline_indices ?? []).join(' / ')}
                  </Typography.Text>
                ) : null}
              </Space>
            ))}
          </Space>
        </div>
      ) : null}
    </Space>
  )
}

export function AuditWorkbench({ workbench }: AuditWorkbenchProps) {
  const [currentPage, setCurrentPage] = useState<number>(1)
  const [selectedStructuralId, setSelectedStructuralId] = useState<string | null>(null)
  const [pendingRuleStructuralId, setPendingRuleStructuralId] = useState<string | null>(null)
  const [activeRuleNavigationAttempt, setActiveRuleNavigationAttempt] = useState<ActiveRuleNavigationAttempt | null>(null)
  const [selectedTocSequenceId, setSelectedTocSequenceId] = useState<string | null>(null)
  const [structureAuditSeverityFilter, setStructureAuditSeverityFilter] = useState<'warnings' | 'passes' | 'all'>(
    'all',
  )
  const [structureAuditIssueFilter, setStructureAuditIssueFilter] = useState<string>('all')
  const [structureAuditFilenameFilter, setStructureAuditFilenameFilter] = useState<string>('all')
  const [focusedRuleId, setFocusedRuleId] = useState<string | null>(null)
  const [ruleCheckStatusFilter, setRuleCheckStatusFilter] = useState<RuleCheckStatusFilter>(
    DEFAULT_RULE_CHECK_STATUS_FILTER,
  )
  const [activeRuleGroupStatusFilter, setActiveRuleGroupStatusFilter] =
    useState<ActiveRuleGroupStatusFilter | null>(null)
  const [activeStructureAuditPathFocus, setActiveStructureAuditPathFocus] = useState<ActiveStructureAuditPathFocus | null>(null)

  const pdfDocument = workbench?.pdf_document
  const reviewPanelVisibility = useMemo(() => getReviewPanelVisibility(workbench), [workbench])
  const singleFileReviewSummary = useMemo(
    () => buildSingleFileReviewSummary(workbench),
    [workbench],
  )
  const showDemoPanels = reviewPanelVisibility.showDemoPanels
  const showSingleFileSummary = reviewPanelVisibility.showSingleFileSummary
  const showProjectReadinessPanels = reviewPanelVisibility.showProjectReadinessPanels
  const navigationAuditRecords = useMemo(
    () => workbench?.rule_checks.navigation_audit_records ?? [],
    [workbench?.rule_checks.navigation_audit_records],
  )
  const structureAuditRecords = useMemo(
    () => workbench?.rule_checks.structure_audit_records ?? [],
    [workbench?.rule_checks.structure_audit_records],
  )
  const structureAuditRecordSignature = useMemo(
    () => buildStructureAuditRecordSignature(structureAuditRecords, { pdfFileId: pdfDocument?.file_id ?? null }),
    [pdfDocument?.file_id, structureAuditRecords],
  )
  const structureAuditIssueOptions = useMemo(() => getStructureAuditIssueOptions(), [])
  const structureAuditFilenameOptions = useMemo(
    () => getStructureAuditFilenameOptions(structureAuditRecords),
    [structureAuditRecords],
  )
  const structureAuditIssueSummary = useMemo(
    () =>
      buildStructureAuditIssueSummary(structureAuditRecords, {
        severityFilter: structureAuditSeverityFilter,
        issueFilter: structureAuditIssueFilter,
        filenameFilter: structureAuditFilenameFilter,
      }),
    [
      structureAuditFilenameFilter,
      structureAuditIssueFilter,
      structureAuditRecords,
      structureAuditSeverityFilter,
    ],
  )
  const structureAuditGroups = useMemo(
    () =>
      buildStructureAuditGroups(structureAuditRecords, {
        severityFilter: structureAuditSeverityFilter,
        issueFilter: structureAuditIssueFilter,
        filenameFilter: structureAuditFilenameFilter,
      }),
    [structureAuditFilenameFilter, structureAuditIssueFilter, structureAuditRecords, structureAuditSeverityFilter],
  )
  const visibleStructureAuditRecordCount = useMemo(
    () => structureAuditGroups.reduce((total, group) => total + group.records.length, 0),
    [structureAuditGroups],
  )
  useEffect(() => {
    setStructureAuditSeverityFilter('all')
    setStructureAuditIssueFilter('all')
    setStructureAuditFilenameFilter('all')
    setActiveStructureAuditPathFocus(null)
  }, [structureAuditRecordSignature])
  const ruleGroupSummaries = useMemo(
    () => workbench?.rule_checks.group_summaries ?? [],
    [workbench?.rule_checks.group_summaries],
  )
  const rawRuleCheckItems = useMemo(() => workbench?.rule_checks.items ?? [], [workbench?.rule_checks.items])
  useEffect(() => {
    setRuleCheckStatusFilter(DEFAULT_RULE_CHECK_STATUS_FILTER)
    setFocusedRuleId(null)
    setActiveRuleGroupStatusFilter(null)
  }, [rawRuleCheckItems])
  const activeRuleGroupSummary = useMemo(
    () =>
      activeRuleGroupStatusFilter
        ? (ruleGroupSummaries.find((group) => group.group_id === activeRuleGroupStatusFilter.groupId) ?? null)
        : null,
    [activeRuleGroupStatusFilter, ruleGroupSummaries],
  )
  const activeRuleGroupScopedRuleIds = activeRuleGroupSummary?.rule_ids ?? null
  const visibleRuleCheckItems = useMemo(
    () => filterRuleCheckItems(rawRuleCheckItems, ruleCheckStatusFilter, activeRuleGroupScopedRuleIds),
    [activeRuleGroupScopedRuleIds, rawRuleCheckItems, ruleCheckStatusFilter],
  )
  const ruleCheckItems = useMemo(
    () => buildStructureAuditRuleFocusList(visibleRuleCheckItems, focusedRuleId),
    [focusedRuleId, visibleRuleCheckItems],
  )
  const ruleCheckStatusFilterOptions = useMemo(
    () => getRuleCheckStatusFilterOptions(workbench?.rule_checks.summary),
    [workbench?.rule_checks.summary],
  )
  const regulatoryReadinessRows = useMemo(
    () => buildRegulatoryReadinessMatrixRows(workbench?.regulatory_readiness?.sources ?? []),
    [workbench?.regulatory_readiness?.sources],
  )
  const ruleItemsById = useMemo(
    () => new Map(rawRuleCheckItems.map((item) => [item.rule_id, item])),
    [rawRuleCheckItems],
  )
  const documentTocSequences = useMemo(
    () => pdfDocument?.toc_sequences ?? [],
    [pdfDocument?.toc_sequences],
  )
  const primaryTocSequenceId = useMemo(
    () => documentTocSequences[0]?.toc_sequence_id ?? null,
    [documentTocSequences],
  )
  const pageOptions = useMemo(
    () =>
      (pdfDocument?.pages ?? []).map((page) => ({
        label: `Page ${page.page_number}`,
        value: page.page_number,
      })),
    [pdfDocument?.pages],
  )

  const scrollToRuleCheckList = () => {
    window.setTimeout(() => {
      document.getElementById('rule-check-list')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 0)
  }

  const focusRule = (ruleId: string) => {
    const ruleItem = ruleItemsById.get(ruleId)
    const statusFilter = ruleItem?.status
    if (statusFilter === 'fail' || statusFilter === 'warn' || statusFilter === 'pass' || statusFilter === 'na') {
      setRuleCheckStatusFilter(statusFilter)
    } else {
      setRuleCheckStatusFilter(DEFAULT_RULE_CHECK_STATUS_FILTER)
    }
    setActiveRuleGroupStatusFilter(null)
    setFocusedRuleId(ruleId)
  }

  const handleRuleSummaryFilterClick = (filterKey: RuleCheckStatusFilter) => {
    setRuleCheckStatusFilter(filterKey)
    setFocusedRuleId(null)
    setActiveRuleGroupStatusFilter(null)
    scrollToRuleCheckList()
  }

  const handleRuleGroupStatusFilterClick = (groupId: string, status: RuleCheckStatusFilter) => {
    setRuleCheckStatusFilter(status)
    setFocusedRuleId(null)
    setActiveRuleGroupStatusFilter({ groupId, status })
    scrollToRuleCheckList()
  }

  const clearRuleGroupStatusFilter = () => {
    setActiveRuleGroupStatusFilter(null)
  }

  const renderStructureAuditGroups = () => {
    if (structureAuditGroups.length === 0) {
      return (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            structureAuditRecords.length > 0
              ? (
                <Space direction="vertical" size={8}>
                  <Typography.Text type="secondary">
                    当前筛选条件隐藏了结构审计记录，请恢复为全部后查看。
                  </Typography.Text>
                  <Button
                    size="small"
                    onClick={() => {
                      setStructureAuditSeverityFilter('all')
                      setStructureAuditIssueFilter('all')
                      setStructureAuditFilenameFilter('all')
                    }}
                  >
                    查看全部结构审计
                  </Button>
                </Space>
              )
              : '暂无结构审计记录'
          }
        />
      )
    }

    return (
              <Space direction="vertical" style={{ width: '100%' }} size={10}>
                {structureAuditGroups.map((group) => (
                  <div key={group.filename} className="structure-audit-group">
                    <Space direction="vertical" style={{ width: '100%' }} size={8}>
                      <Space wrap>
                        <Button
                          size="small"
                          type={structureAuditFilenameFilter === group.filename ? 'primary' : 'default'}
                          onClick={() =>
                            setStructureAuditFilenameFilter(
                              toggleStructureAuditFilterValue(structureAuditFilenameFilter, group.filename),
                            )
                          }
                        >
                          {group.filename}
                        </Button>
                        {group.warnCount > 0 ? <Tag color="orange">预警 {group.warnCount}</Tag> : null}
                        {group.passCount > 0 ? <Tag color="green">通过 {group.passCount}</Tag> : null}
                        <Tag>记录 {group.records.length}</Tag>
                      </Space>
                      {group.issueKeys.length > 0 ? (
                        <Space size={[8, 8]} wrap>
                          {group.issueKeys.map((issueKey) => (
                            <Tag
                              key={`${group.filename}:${issueKey}`}
                              color={structureAuditIssueFilter === issueKey ? 'magenta' : 'orange'}
                              className="clickable-tag"
                              onClick={() =>
                                setStructureAuditIssueFilter(
                                  toggleStructureAuditFilterValue(structureAuditIssueFilter, issueKey),
                                )
                              }
                            >
                              {getStructureAuditIssueLabel(issueKey)}
                            </Tag>
                          ))}
                        </Space>
              ) : (
                <Typography.Text type="secondary" className="rule-detail-line">
                  当前筛选下未发现结构预警问题。
                </Typography.Text>
              )}
              <List
                size="small"
                bordered
                dataSource={group.records}
                renderItem={(record: RuleStructureAuditRecord) => {
                  const directChildPathRows = getStructureAuditMissingPathRows(record, 'direct_child')
                  const boundedSubtreePathRows = getStructureAuditMissingPathRows(record, 'bounded_subtree')
                  const alignmentPathRows = getStructureAuditPathAlignmentRows(record)
                  const matchedAlignmentCount = alignmentPathRows.filter((row) => row.hasBodyMatch).length
                  const reviewAlignmentCount = alignmentPathRows.length - matchedAlignmentCount

                  return (
                  <List.Item className="rule-check-item">
                    <Space direction="vertical" style={{ width: '100%' }} size={4}>
                      <Space wrap>
                        <Tag color={record.alignment_ready ? 'green' : 'orange'}>
                          {record.alignment_ready ? '结构通过' : '结构预警'}
                        </Tag>
                        <Typography.Text type="secondary">{record.rule_id}</Typography.Text>
                      </Space>
                      <Space size={[8, 8]} wrap>
                        {formatRatio(record.root_outline_coverage_ratio) ? (
                          <Tag>根章节覆盖 {formatRatio(record.root_outline_coverage_ratio)}</Tag>
                        ) : null}
                        {formatRatio(record.direct_child_coverage_ratio) ? (
                          <Tag>直接子级覆盖 {formatRatio(record.direct_child_coverage_ratio)}</Tag>
                        ) : null}
                        {formatRatio(record.bounded_subtree_coverage_ratio) ? (
                          <Tag>子树覆盖 {formatRatio(record.bounded_subtree_coverage_ratio)}</Tag>
                        ) : null}
                        {record.root_page_order_ready === false ? (
                          <Tag color="orange">根章节页序冲突</Tag>
                        ) : null}
                        {record.root_page_offset_ready === false ? (
                          <Tag color="orange">根章节页码偏移冲突</Tag>
                        ) : null}
                        {record.root_page_span_ready === false ? (
                          <Tag color="orange">根章节页区间冲突</Tag>
                        ) : null}
                        <Button size="small" type="link" onClick={() => focusRule(record.rule_id)}>
                          定位规则
                        </Button>
                        {(
                          (record.navigation_targets ?? []).length > 0
                            ? (record.navigation_targets ?? []).map((target) => ({
                                key: target.target_kind,
                                label: target.label,
                                page: target.page,
                                tocSequenceId: target.toc_sequence_id ?? null,
                              }))
                            : buildStructureAuditNavigationTargets(record, {
                                activeFilename: pdfDocument?.filename ?? null,
                                tocSequences: documentTocSequences,
                              })
                        ).map((target) => (
                          <Button
                            key={`${record.audit_record_id ?? record.rule_id}-${target.key}`}
                            size="small"
                            type="link"
                            onClick={() => {
                              setCurrentPage(target.page)
                              if (target.tocSequenceId) {
                                setSelectedTocSequenceId(target.tocSequenceId)
                              }
                              focusRule(record.rule_id)
                            }}
                          >
                            {target.label}
                          </Button>
                        ))}
                      </Space>
                      {(record.missing_body_root_outline_indices ?? []).length > 0 ? (
                        <Typography.Text type="secondary" className="rule-detail-line">
                          缺失根章节: {(record.missing_body_root_outline_indices ?? []).join(' / ')}
                        </Typography.Text>
                      ) : null}
                      {(record.missing_body_direct_child_outline_indices ?? []).length > 0 ? (
                        <Typography.Text type="secondary" className="rule-detail-line">
                          缺失直接子级: {(record.missing_body_direct_child_outline_indices ?? []).join(' / ')}
                        </Typography.Text>
                      ) : null}
                      {directChildPathRows.length > 0 ? (
                        <div className="rule-detail-block">
                          <Typography.Text type="secondary" className="rule-detail-line">
                            缺失直接子级路径:
                          </Typography.Text>
                          {directChildPathRows.map((pathRow, index) => {
                            const focusKey = buildStructureAuditPathFocusKey(pathRow, 'direct_child')
                            const isActive = activeStructureAuditPathFocus?.key === focusKey
                            return (
                            <Space
                              key={`${record.audit_record_id ?? record.rule_id}-direct-path-${index}`}
                              size={[8, 4]}
                              wrap
                              className={[
                                'rule-detail-line',
                                'rule-detail-path-row',
                                isActive ? 'rule-detail-path-row-active' : '',
                              ]
                                .filter(Boolean)
                                .join(' ')}
                            >
                              <Typography.Text type="secondary" className="rule-detail-path-line">
                                {formatStructureAuditPathRow(pathRow)}
                              </Typography.Text>
                              {typeof pathRow.navigationPage === 'number' ? (
                                <Button
                                  size="small"
                                  type="link"
                                  onClick={() => {
                                    const target = resolveStructureAuditPathNavigationTarget(pathRow, 'toc', {
                                      tocBlocks: pdfDocument?.toc_blocks ?? [],
                                      preferredTocSequenceId: primaryTocSequenceId,
                                    })
                                    if (target.tocSequenceId) {
                                      setSelectedTocSequenceId(target.tocSequenceId)
                                    }
                                    setActiveStructureAuditPathFocus({
                                      key: focusKey,
                                      summary: buildStructureAuditPathFocusLabel(pathRow, 'toc'),
                                    })
                                    handleRuleNavigation(
                                      target.page ?? undefined,
                                      target.structuralId ?? undefined,
                                      `结构缺失路径 / ${pathRow.textPath || pathRow.outlinePath || pathRow.outlineIndex || 'TOC'}`,
                                      target.navigationStatus,
                                      target.navigationReason,
                                      null,
                                    )
                                    focusRule(record.rule_id)
                                  }}
                                >
                                  TOC
                                </Button>
                              ) : null}
                              {typeof pathRow.nearestBodyAnchorPage === 'number' ? (
                                <Button
                                  size="small"
                                  type="link"
                                  onClick={() => {
                                    const target = resolveStructureAuditPathNavigationTarget(pathRow, 'body')
                                    setActiveStructureAuditPathFocus({
                                      key: focusKey,
                                      summary: buildStructureAuditPathFocusLabel(pathRow, 'body'),
                                    })
                                    handleRuleNavigation(
                                      target.page ?? undefined,
                                      undefined,
                                      `最近正文锚点 / ${pathRow.nearestBodyAnchorOutlineIndex || pathRow.parentOutlineIndex || pathRow.rootOutlineIndex || '正文'}`,
                                      target.navigationStatus,
                                      target.navigationReason,
                                      null,
                                    )
                                    focusRule(record.rule_id)
                                  }}
                                >
                                  正文
                                </Button>
                              ) : null}
                            </Space>
                            )
                          })}
                        </div>
                      ) : null}
                      {(record.missing_body_bounded_subtree_outline_indices ?? []).length > 0 ? (
                        <Typography.Text type="secondary" className="rule-detail-line">
                          缺失子树节点: {(record.missing_body_bounded_subtree_outline_indices ?? []).join(' / ')}
                        </Typography.Text>
                      ) : null}
                      {boundedSubtreePathRows.length > 0 ? (
                        <div className="rule-detail-block">
                          <Typography.Text type="secondary" className="rule-detail-line">
                            缺失子树路径:
                          </Typography.Text>
                          {boundedSubtreePathRows.map((pathRow, index) => {
                            const focusKey = buildStructureAuditPathFocusKey(pathRow, 'bounded_subtree')
                            const isActive = activeStructureAuditPathFocus?.key === focusKey
                            return (
                            <Space
                              key={`${record.audit_record_id ?? record.rule_id}-subtree-path-${index}`}
                              size={[8, 4]}
                              wrap
                              className={[
                                'rule-detail-line',
                                'rule-detail-path-row',
                                isActive ? 'rule-detail-path-row-active' : '',
                              ]
                                .filter(Boolean)
                                .join(' ')}
                            >
                              <Typography.Text type="secondary" className="rule-detail-path-line">
                                {formatStructureAuditPathRow(pathRow)}
                              </Typography.Text>
                              {typeof pathRow.navigationPage === 'number' ? (
                                <Button
                                  size="small"
                                  type="link"
                                  onClick={() => {
                                    const target = resolveStructureAuditPathNavigationTarget(pathRow, 'toc', {
                                      tocBlocks: pdfDocument?.toc_blocks ?? [],
                                      preferredTocSequenceId: primaryTocSequenceId,
                                    })
                                    if (target.tocSequenceId) {
                                      setSelectedTocSequenceId(target.tocSequenceId)
                                    }
                                    setActiveStructureAuditPathFocus({
                                      key: focusKey,
                                      summary: buildStructureAuditPathFocusLabel(pathRow, 'toc'),
                                    })
                                    handleRuleNavigation(
                                      target.page ?? undefined,
                                      target.structuralId ?? undefined,
                                      `结构缺失路径 / ${pathRow.textPath || pathRow.outlinePath || pathRow.outlineIndex || 'TOC'}`,
                                      target.navigationStatus,
                                      target.navigationReason,
                                      null,
                                    )
                                    focusRule(record.rule_id)
                                  }}
                                >
                                  TOC
                                </Button>
                              ) : null}
                              {typeof pathRow.nearestBodyAnchorPage === 'number' ? (
                                <Button
                                  size="small"
                                  type="link"
                                  onClick={() => {
                                    const target = resolveStructureAuditPathNavigationTarget(pathRow, 'body')
                                    setActiveStructureAuditPathFocus({
                                      key: focusKey,
                                      summary: buildStructureAuditPathFocusLabel(pathRow, 'body'),
                                    })
                                    handleRuleNavigation(
                                      target.page ?? undefined,
                                      undefined,
                                      `最近正文锚点 / ${pathRow.nearestBodyAnchorOutlineIndex || pathRow.parentOutlineIndex || pathRow.rootOutlineIndex || '正文'}`,
                                      target.navigationStatus,
                                      target.navigationReason,
                                      null,
                                    )
                                    focusRule(record.rule_id)
                                  }}
                                >
                                  正文
                                </Button>
                              ) : null}
                            </Space>
                            )
                          })}
                        </div>
                      ) : null}
                      {alignmentPathRows.length > 0 ? (
                        <div className="structure-audit-path-table-block">
                          <Space wrap className="structure-audit-path-table-title">
                            <Typography.Text strong>目录-正文章节定位</Typography.Text>
                            <Tag>目录条目 {alignmentPathRows.length}</Tag>
                            <Tag color="green">已匹配 {matchedAlignmentCount}</Tag>
                            {reviewAlignmentCount > 0 ? <Tag color="orange">需核对 {reviewAlignmentCount}</Tag> : null}
                          </Space>
                          <div className="structure-audit-path-table">
                            <div className="structure-audit-path-table-row structure-audit-path-table-header">
                              <span>章节</span>
                              <span>标题路径</span>
                              <span>TOC页</span>
                              <span>标注页</span>
                              <span>正文页</span>
                              <span>状态</span>
                              <span>操作</span>
                            </div>
                            {alignmentPathRows.map((pathRow, index) => {
                              const focusKey = buildStructureAuditPathFocusKey(pathRow, 'toc_body')
                              const isActive = activeStructureAuditPathFocus?.key === focusKey
                              return (
                                <div
                                  key={`${record.audit_record_id ?? record.rule_id}-toc-body-path-${index}`}
                                  className={[
                                    'structure-audit-path-table-row',
                                    isActive ? 'structure-audit-path-table-row-active' : '',
                                  ]
                                    .filter(Boolean)
                                    .join(' ')}
                                >
                                  <span>{pathRow.outlineIndex || '-'}</span>
                                  <span title={pathRow.textPath || pathRow.outlinePath || undefined}>
                                    {pathRow.textPath || pathRow.outlinePath || '-'}
                                  </span>
                                  <span>{pathRow.navigationPage ?? '-'}</span>
                                  <span>{pathRow.pageLocatorValue ?? '-'}</span>
                                  <span>{pathRow.bodyAnchorPage ?? '-'}</span>
                                  <span>{getStructureAuditAlignmentStatusLabel(pathRow.alignmentStatus)}</span>
                                  <span>
                                    <Space size={4} wrap>
                                      {typeof pathRow.navigationPage === 'number' ? (
                                        <Button
                                          size="small"
                                          type="link"
                                          onClick={() => {
                                            const target = resolveStructureAuditPathNavigationTarget(pathRow, 'toc', {
                                              tocBlocks: pdfDocument?.toc_blocks ?? [],
                                              preferredTocSequenceId: primaryTocSequenceId,
                                            })
                                            if (target.tocSequenceId) {
                                              setSelectedTocSequenceId(target.tocSequenceId)
                                            }
                                            setActiveStructureAuditPathFocus({
                                              key: focusKey,
                                              summary: buildStructureAuditPathFocusLabel(pathRow, 'toc'),
                                            })
                                            handleRuleNavigation(
                                              target.page ?? undefined,
                                              target.structuralId ?? undefined,
                                              `目录路径 / ${pathRow.textPath || pathRow.outlinePath || pathRow.outlineIndex || 'TOC'}`,
                                              target.navigationStatus,
                                              target.navigationReason,
                                              null,
                                            )
                                            focusRule(record.rule_id)
                                          }}
                                        >
                                          TOC
                                        </Button>
                                      ) : null}
                                      {typeof pathRow.bodyAnchorPage === 'number' ? (
                                        <Button
                                          size="small"
                                          type="link"
                                          onClick={() => {
                                            const target = resolveStructureAuditPathNavigationTarget(pathRow, 'body')
                                            setActiveStructureAuditPathFocus({
                                              key: focusKey,
                                              summary: buildStructureAuditPathFocusLabel(pathRow, 'body'),
                                            })
                                            handleRuleNavigation(
                                              target.page ?? undefined,
                                              undefined,
                                              `正文锚点 / ${pathRow.bodyAnchorOutlineIndex || pathRow.parentOutlineIndex || pathRow.rootOutlineIndex || '正文'}`,
                                              target.navigationStatus,
                                              target.navigationReason,
                                              null,
                                            )
                                            focusRule(record.rule_id)
                                          }}
                                        >
                                          正文
                                        </Button>
                                      ) : null}
                                    </Space>
                                  </span>
                                </div>
                              )
                            })}
                          </div>
                        </div>
                      ) : null}
                      {(record.root_page_alignment_rows ?? []).length > 0 ? (
                        <div className="structure-audit-alignment-list">
                          <Typography.Text strong>根章节冲突明细</Typography.Text>
                          {getStructureAuditRootPageConflictRows(record).length > 0 ? (
                            <div className="structure-audit-alignment-table">
                              <div className="structure-audit-alignment-table-row structure-audit-alignment-table-header">
                                <span>章节</span>
                                <span>TOC所在页</span>
                                <span>TOC标注页</span>
                                <span>正文页区间</span>
                                <span>投影页</span>
                                <span>偏移</span>
                                <span>冲突类型</span>
                                <span>操作</span>
                              </div>
                              {getStructureAuditRootPageConflictRows(record).map((alignmentRow, index) => (
                                <div
                                  key={`${record.audit_record_id ?? record.rule_id}-root-align-${index}`}
                                  className="structure-audit-alignment-table-row"
                                >
                                  <span>{alignmentRow.outline_index}</span>
                                  <span>{alignmentRow.toc_navigation_page ?? '-'}</span>
                                  <span>{alignmentRow.toc_page_locator_value}</span>
                                  <span>
                                    {alignmentRow.body_page_start}-{alignmentRow.body_page_end}
                                  </span>
                                  <span>{typeof alignmentRow.projected_page === 'number' ? alignmentRow.projected_page : '-'}</span>
                                  <span>{typeof alignmentRow.offset === 'number' ? alignmentRow.offset : '-'}</span>
                                  <span>
                                    {[
                                      alignmentRow.order_conflict ? '页序冲突' : null,
                                      alignmentRow.offset_conflict ? '偏移冲突' : null,
                                      alignmentRow.span_conflict ? '区间冲突' : null,
                                    ]
                                      .filter(Boolean)
                                      .join(' / ')}
                                  </span>
                                  <span>
                                    <Space size={4} wrap>
                                      <Button
                                        size="small"
                                        type="link"
                                        onClick={() => {
                                          setCurrentPage(alignmentRow.toc_navigation_page ?? alignmentRow.toc_page_locator_value)
                                          if (primaryTocSequenceId) {
                                            setSelectedTocSequenceId(primaryTocSequenceId)
                                          }
                                          focusRule(record.rule_id)
                                        }}
                                      >
                                        TOC
                                      </Button>
                                      <Button
                                        size="small"
                                        type="link"
                                        onClick={() => {
                                          setCurrentPage(alignmentRow.body_page_start)
                                          focusRule(record.rule_id)
                                        }}
                                      >
                                        正文
                                      </Button>
                                    </Space>
                                  </span>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <Typography.Text type="secondary" className="rule-detail-line">
                              根章节页码映射已通过，共 {(record.root_page_alignment_rows ?? []).length} 个根章节完成校验。
                            </Typography.Text>
                          )}
                        </div>
                      ) : null}
                    </Space>
                  </List.Item>
                  )
                }}
              />
            </Space>
          </div>
        ))}
      </Space>
    )
  }

  const getSingleFileModuleTagColor = (status: string) => {
    switch (status) {
      case 'pass':
        return 'green'
      case 'fail':
        return 'red'
      case 'review_required':
        return 'orange'
      case 'not_available':
        return 'default'
      default:
        return 'blue'
    }
  }

  const getSingleFileModuleStatusLabel = (status: string) => {
    switch (status) {
      case 'pass':
        return '未见重点风险'
      case 'fail':
        return '需优先处理'
      case 'review_required':
        return '建议复核'
      case 'not_available':
        return '暂不适用'
      default:
        return status
    }
  }

  useEffect(() => {
    if (pageOptions.length === 0) {
      return
    }
    if (pageOptions.some((page) => page.value === currentPage)) {
      return
    }
    setCurrentPage(pageOptions[0].value)
  }, [currentPage, pageOptions])

  useEffect(() => {
    if (!focusedRuleId) {
      return
    }
    const target = document.getElementById(`rule-check-item-${focusedRuleId}`)
    target?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [focusedRuleId, ruleCheckItems])

  const activePage = useMemo(
    () => pdfDocument?.pages.find((page) => page.page_number === currentPage) ?? null,
    [pdfDocument?.pages, currentPage],
  )

  const activePageBoundingBoxes = useMemo(
    () => (pdfDocument?.bounding_boxes ?? []).filter((item) => item.page_number === currentPage),
    [pdfDocument?.bounding_boxes, currentPage],
  )

  const activePageStructuralBoxes = useMemo(
    () => activePageBoundingBoxes.filter(isStructuralBoundingBox),
    [activePageBoundingBoxes],
  )

  const activePageTables = useMemo(
    () => (pdfDocument?.table_asts ?? []).filter((table) => table.page === currentPage),
    [pdfDocument?.table_asts, currentPage],
  )

  const activePageImages = useMemo(
    () => (pdfDocument?.image_blocks ?? []).filter((image) => image.page === currentPage),
    [pdfDocument?.image_blocks, currentPage],
  )

  const activePageAlgorithms = useMemo(
    () => (pdfDocument?.algorithm_blocks ?? []).filter((algorithm) => algorithm.page === currentPage),
    [pdfDocument?.algorithm_blocks, currentPage],
  )

  const activePageEquations = useMemo(
    () => (pdfDocument?.equation_blocks ?? []).filter((equation) => equation.page === currentPage),
    [pdfDocument?.equation_blocks, currentPage],
  )

  const activePageTocBlocks = useMemo(
    () => (pdfDocument?.toc_blocks ?? []).filter((toc) => toc.page === currentPage),
    [pdfDocument?.toc_blocks, currentPage],
  )

  const selectedAlgorithm = useMemo(
    () =>
      activePageAlgorithms.find((algorithm) => algorithm.algorithm_id === selectedStructuralId) ??
      null,
    [activePageAlgorithms, selectedStructuralId],
  )

  const activePageCompactedTables = useMemo(
    () => activePageTables.filter((table) => table.semantic_compaction?.applied),
    [activePageTables],
  )

  const tocSequenceLookup = useMemo(() => {
    const lookup = new Map<string, PdfTocSequence>()
    documentTocSequences.forEach((sequence) => {
      lookup.set(sequence.toc_sequence_id, sequence)
    })
    return lookup
  }, [documentTocSequences])

  const activePageTocSequenceIds = useMemo(() => {
    const ids = new Set<string>()
    activePageTocBlocks.forEach((toc) => {
      if (toc.toc_sequence_id) {
        ids.add(toc.toc_sequence_id)
      }
    })
    documentTocSequences.forEach((sequence) => {
      if ((sequence.pages ?? []).includes(currentPage)) {
        ids.add(sequence.toc_sequence_id)
      }
    })
    return Array.from(ids)
  }, [activePageTocBlocks, currentPage, documentTocSequences])

  const activePageTocSequences = useMemo(
    () =>
      documentTocSequences.filter((sequence) =>
        activePageTocSequenceIds.includes(sequence.toc_sequence_id),
      ),
    [activePageTocSequenceIds, documentTocSequences],
  )

  useEffect(() => {
    const sequenceIds = documentTocSequences.map((sequence) => sequence.toc_sequence_id)
    if (sequenceIds.length === 0) {
      setSelectedTocSequenceId(null)
      return
    }

    if (selectedTocSequenceId && sequenceIds.includes(selectedTocSequenceId)) {
      return
    }

    if (activePageTocSequenceIds.length > 0) {
      setSelectedTocSequenceId(activePageTocSequenceIds[0])
      return
    }

    setSelectedTocSequenceId(sequenceIds[0])
  }, [activePageTocSequenceIds, documentTocSequences, selectedTocSequenceId])

  const selectedTocSequence = useMemo(
    () =>
      documentTocSequences.find((sequence) => sequence.toc_sequence_id === selectedTocSequenceId) ??
      null,
    [documentTocSequences, selectedTocSequenceId],
  )

  const activePageStructuralLookup = useMemo(() => {
    const lookup = new Map<string, string>()
    activePageTables.forEach((table) => {
      lookup.set(
        table.table_id,
        table.title?.trim() ? `${table.table_id} | ${table.title}` : table.table_id,
      )
    })
    activePageImages.forEach((image) => {
      lookup.set(
        image.image_id,
        image.title?.trim() ? `${image.image_id} | ${image.title}` : image.image_id,
      )
    })
    activePageAlgorithms.forEach((algorithm) => {
      const title = buildAlgorithmTitle(algorithm)
      lookup.set(
        algorithm.algorithm_id,
        title ? `${algorithm.algorithm_id} | ${title}` : algorithm.algorithm_id,
      )
    })
    activePageEquations.forEach((equation) => {
      lookup.set(
        equation.equation_id,
        equation.text?.trim() ? `${equation.equation_id} | ${equation.text}` : equation.equation_id,
      )
    })
    activePageTocBlocks.forEach((toc) => {
      lookup.set(toc.toc_id, toc.title?.trim() ? `${toc.toc_id} | ${toc.title}` : toc.toc_id)
    })
    return lookup
  }, [activePageAlgorithms, activePageEquations, activePageImages, activePageTables, activePageTocBlocks])

  const activePageStructuralSelectionOrder = useMemo(() => {
    const orderedIds = [
      ...activePageTables.map((table) => table.table_id),
      ...activePageImages.map((image) => image.image_id),
      ...activePageAlgorithms.map((algorithm) => algorithm.algorithm_id),
      ...activePageEquations.map((equation) => equation.equation_id),
      ...activePageTocBlocks.map((toc) => toc.toc_id),
    ]
    activePageStructuralBoxes.forEach((item) => {
      if (!orderedIds.includes(item.id)) {
        orderedIds.push(item.id)
      }
    })
    return orderedIds
  }, [activePageAlgorithms, activePageEquations, activePageImages, activePageStructuralBoxes, activePageTables, activePageTocBlocks])

  useEffect(() => {
    const resolution = resolveStructuralSelection({
      activePageStructuralSelectionOrder,
      selectedStructuralId,
      pendingStructuralId: pendingRuleStructuralId,
    })

    if (resolution.selectedStructuralId !== selectedStructuralId) {
      setSelectedStructuralId(resolution.selectedStructuralId)
    }
    if (resolution.pendingStructuralId !== pendingRuleStructuralId) {
      setPendingRuleStructuralId(resolution.pendingStructuralId)
    }
  }, [activePageStructuralSelectionOrder, pendingRuleStructuralId, selectedStructuralId])

  const activePageBoundingBoxesOrdered = useMemo(() => {
    const nonSelected = activePageBoundingBoxes.filter((item) => item.id !== selectedStructuralId)
    const selected = activePageBoundingBoxes.filter((item) => item.id === selectedStructuralId)
    return [...nonSelected, ...selected]
  }, [activePageBoundingBoxes, selectedStructuralId])

  const activeBoundingBoxListItems = useMemo(() => {
    const structuralBoxes = activePageBoundingBoxes.filter(isStructuralBoundingBox)
    const textBoxes = activePageBoundingBoxes.filter((item) => !isStructuralBoundingBox(item))
    return [...structuralBoxes, ...textBoxes.slice(0, TEXT_BOUNDING_BOX_LIST_LIMIT)]
  }, [activePageBoundingBoxes])

  const ruleNavigationLanding = useMemo(
    () =>
      resolveRuleNavigationLanding({
        currentPage,
        activePageStructuralSelectionOrder,
        activePageStructuralBoundingBoxIds: activePageStructuralBoxes.map((item) => item.id),
        selectedStructuralId,
        targetPage: activeRuleNavigationAttempt?.targetPage ?? null,
        targetStructuralId: activeRuleNavigationAttempt?.targetStructuralId ?? null,
      }),
    [activePageStructuralBoxes, activePageStructuralSelectionOrder, activeRuleNavigationAttempt, currentPage, selectedStructuralId],
  )

  const endToEndRuleNavigationVerification = useMemo(
    () =>
      resolveEndToEndRuleNavigationVerification({
        backendNavigationStatus: activeRuleNavigationAttempt?.backendNavigationStatus ?? null,
        backendNavigationReason: activeRuleNavigationAttempt?.backendNavigationReason ?? null,
        landingStatus: ruleNavigationLanding.status,
        landingReason: ruleNavigationLanding.reason,
      }),
    [activeRuleNavigationAttempt, ruleNavigationLanding.reason, ruleNavigationLanding.status],
  )

  const activeRuleNavigationAuditRecord = useMemo<EndToEndNavigationAuditChain | null>(
    () =>
      activeRuleNavigationAttempt
        ? buildEndToEndNavigationAuditChain({
            backendAuditRecord: activeRuleNavigationAttempt.backendAuditRecord,
            landingStatus: ruleNavigationLanding.status,
            landingReason: ruleNavigationLanding.reason,
            verificationStatus: endToEndRuleNavigationVerification.status,
            verificationReason: endToEndRuleNavigationVerification.reason,
          })
        : null,
    [activeRuleNavigationAttempt, endToEndRuleNavigationVerification.reason, endToEndRuleNavigationVerification.status, ruleNavigationLanding.reason, ruleNavigationLanding.status],
  )

  function handleRuleNavigation(
    page?: number,
    structuralId?: string,
    label?: string,
    backendNavigationStatus?: string,
    backendNavigationReason?: string,
    backendAuditRecord?: RuleNavigationAuditRecord | null,
  ) {
    const normalizedStructuralId = structuralId?.trim() || null
    const normalizedPage = typeof page === 'number' && page > 0 ? page : null
    setActiveRuleNavigationAttempt({
      label: label?.trim() || 'rule-navigation',
      targetPage: normalizedPage,
      targetStructuralId: normalizedStructuralId,
      backendNavigationStatus: backendNavigationStatus?.trim() || null,
      backendNavigationReason: backendNavigationReason?.trim() || null,
      backendAuditRecord: backendAuditRecord ?? null,
    })
    if (typeof page === 'number' && page > 0) {
      setCurrentPage(page)
    }
    if (normalizedStructuralId) {
      if (
        typeof page === 'number' &&
        page > 0 &&
        page !== currentPage
      ) {
        setPendingRuleStructuralId(normalizedStructuralId)
        setSelectedStructuralId(null)
      } else {
        setPendingRuleStructuralId(null)
        setSelectedStructuralId(normalizedStructuralId)
      }
    } else {
      setPendingRuleStructuralId(null)
    }
  }

  const ratio = activePage ? PDF_RENDER_WIDTH / activePage.width : 1
  const overlayHeight = activePage ? activePage.height * ratio : 0

  return (
    <ProCard
      title="智能审阅工作台"
      subTitle="查看原文结构、解析摘要、法规来源、前置条件和复核提示。"
      bordered
      headerBordered
    >
      <ProCard split="vertical">
        <ProCard title="PDF 阅读与结构框高亮" colSpan="36%" className="workbench-col">
          {!pdfDocument ? (
            <Empty description="本次上传未包含 PDF，无法显示 PDF 阅读器" />
          ) : (
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Typography.Text strong>{pdfDocument.filename}</Typography.Text>
              <Select
                value={currentPage}
                options={pageOptions}
                style={{ width: 180 }}
                onChange={setCurrentPage}
              />
              <Space size={[8, 8]} wrap>
                <Tag color="blue">当前页框 {activePageBoundingBoxes.length}</Tag>
                <Tag color="geekblue">结构框 {activePageStructuralBoxes.length}</Tag>
                <Tag color="purple">表格 {activePageTables.length}</Tag>
                <Tag color="gold">图片 {activePageImages.length}</Tag>
                <Tag color="orange">算法 {activePageAlgorithms.length}</Tag>
                <Tag color="lime">公式 {activePageEquations.length}</Tag>
                <Tag color="cyan">TOC {activePageTocBlocks.length}</Tag>
                {documentTocSequences.length > 0 ? (
                  <Tag color="processing">TOC 序列 {documentTocSequences.length}</Tag>
                ) : null}
                {activePageTocSequences.length > 0 ? (
                  <Tag color="blue-inverse">当前页序列 {activePageTocSequences.length}</Tag>
                ) : null}
                {activePage?.layout_mode ? <Tag>{activePage.layout_mode}</Tag> : null}
              </Space>
              <div className="pdf-viewer-shell">
                <Document file={buildAssetUrl(pdfDocument.file_url)} loading="加载 PDF 中...">
                  <div
                    className="pdf-page-wrapper"
                    style={{ width: PDF_RENDER_WIDTH, height: overlayHeight || undefined }}
                  >
                    <Page
                      pageNumber={currentPage}
                      width={PDF_RENDER_WIDTH}
                      renderAnnotationLayer={false}
                      renderTextLayer={false}
                    />
                    {activePage ? (
                      <div
                        className="bbox-overlay-layer"
                        style={{ width: PDF_RENDER_WIDTH, height: overlayHeight }}
                      >
                        {activePageBoundingBoxesOrdered.map((item) => {
                          const kind = getBoundingBoxKind(item)
                          const structural = kind !== 'text'
                          const selected = structural && item.id === selectedStructuralId
                          const boxWidth = (item.bbox.x1 - item.bbox.x0) * ratio
                          const boxHeight = (item.bbox.y1 - item.bbox.y0) * ratio
                          return (
                            <div
                              key={item.id}
                              className={[
                                'bbox-rect',
                                structural ? 'bbox-rect-structural' : '',
                                structural ? getBoundingBoxKindClassName(kind) : '',
                                selected ? 'bbox-rect-selected' : '',
                              ]
                                .filter(Boolean)
                                .join(' ')}
                              style={{
                                left: item.bbox.x0 * ratio,
                                top: item.bbox.y0 * ratio,
                                width: boxWidth,
                                height: boxHeight,
                              }}
                              title={activePageStructuralLookup.get(item.id) ?? item.text}
                            >
                              {selected ? (
                                <span className="bbox-rect-label">
                                  {activePageStructuralLookup.get(item.id) ?? item.id}
                                </span>
                              ) : null}
                            </div>
                          )
                        })}
                      </div>
                    ) : null}
                  </div>
                </Document>
              </div>
              {selectedTocSequence ? (
                <Alert
                  type="info"
                  showIcon
                  message={`当前选中 TOC 序列: ${buildTocSequenceTitle(selectedTocSequence)}`}
                  description={
                    <Space direction="vertical" size={2}>
                      <Typography.Text>
                        {buildTocSequenceSummary(selectedTocSequence, currentPage)}
                      </Typography.Text>
                      {buildTocSequenceDetail(selectedTocSequence) ? (
                        <Typography.Text type="secondary">
                          {buildTocSequenceDetail(selectedTocSequence)}
                        </Typography.Text>
                      ) : null}
                    </Space>
                  }
                />
              ) : null}
              {activeStructureAuditPathFocus ? (
                <Alert
                  type="info"
                  showIcon
                  message={`当前定位路径: ${activeStructureAuditPathFocus.summary}`}
                  description="结构审计中的当前激活路径已同步到 PDF 导航状态。"
                />
              ) : null}
              {activeRuleNavigationAttempt && ruleNavigationLanding.status !== 'idle' ? (
                <Alert
                  type={
                    activeRuleNavigationAuditRecord?.severity === 'success'
                      ? 'success'
                      : activeRuleNavigationAuditRecord?.severity === 'warning'
                        ? 'warning'
                        : 'info'
                  }
                  showIcon
                  message={`规则导航落点: ${activeRuleNavigationAttempt.label}`}
                  description={
                    <Space direction="vertical" size={2}>
                      <Typography.Text>
                        {getNavigationStatusLabel(ruleNavigationLanding.status)} |{' '}
                        {getNavigationReasonLabel(ruleNavigationLanding.reason)}
                      </Typography.Text>
                      <Typography.Text type="secondary">
                        端到端验证: {getVerificationStatusLabel(endToEndRuleNavigationVerification.status)} |{' '}
                        {getNavigationReasonLabel(endToEndRuleNavigationVerification.reason)}
                      </Typography.Text>
                      {activeRuleNavigationAuditRecord ? (
                        <Typography.Text type="secondary">
                          审计等级: {activeRuleNavigationAuditRecord.severity} | 终态:{' '}
                          {activeRuleNavigationAuditRecord.isTerminal ? '是' : '否'}
                        </Typography.Text>
                      ) : null}
                      {activeRuleNavigationAttempt.targetPage ? (
                        <Typography.Text type="secondary">
                          target page: {activeRuleNavigationAttempt.targetPage}
                          {activeRuleNavigationAttempt.targetStructuralId
                            ? ` | target structural: ${activeRuleNavigationAttempt.targetStructuralId}`
                            : ''}
                        </Typography.Text>
                      ) : null}
                    </Space>
                  }
                />
              ) : null}
              {activePageCompactedTables.length > 0 ? (
                <Alert
                  type="info"
                  showIcon
                  message={`当前页有 ${activePageCompactedTables.length} 个表格展示的是语义视图；原始行证据仍然保留在解析结果中。点击下方结构对象摘要可联动高亮对应区域。`}
                />
              ) : null}
              {selectedAlgorithm ? (
                <List
                  size="small"
                  bordered
                  header="当前选中算法详情"
                  dataSource={selectedAlgorithm.lines ?? []}
                  renderItem={(line, index) => (
                    <List.Item>
                      <Space direction="vertical" size={2} style={{ width: '100%' }}>
                        {index === 0 ? (
                          <Space size={[8, 8]} wrap>
                            <Tag color="orange">{selectedAlgorithm.algorithm_id}</Tag>
                            {selectedAlgorithm.algorithm_ref ? (
                              <Tag color="blue">{selectedAlgorithm.algorithm_ref}</Tag>
                            ) : null}
                            {selectedAlgorithm.continued_from_previous_page ? (
                              <Tag color="geekblue">上页续接</Tag>
                            ) : null}
                            {selectedAlgorithm.continues_to_next_page ? (
                              <Tag color="gold">续至下页</Tag>
                            ) : null}
                          </Space>
                        ) : null}
                        <Typography.Text>{line}</Typography.Text>
                      </Space>
                    </List.Item>
                  )}
                />
              ) : null}
              {documentTocSequences.length > 0 ? (
                <List
                  size="small"
                  bordered
                  header="文档级 TOC 序列摘要"
                  dataSource={documentTocSequences}
                  renderItem={(sequence) => {
                    const pageLabel = buildTocSequencePageLabel(sequence)
                    const detail = buildTocSequenceDetail(sequence)
                    const isSelected = sequence.toc_sequence_id === selectedTocSequenceId
                    const includesCurrentPage = (sequence.pages ?? []).includes(currentPage)
                    const jumpTarget = sequence.pages?.[0]
                    return (
                      <List.Item
                        className={
                          isSelected
                            ? 'structural-summary-item structural-summary-item-selected'
                            : 'structural-summary-item'
                        }
                        onClick={() => {
                          setSelectedTocSequenceId(sequence.toc_sequence_id)
                          if (
                            typeof jumpTarget === 'number' &&
                            !(sequence.pages ?? []).includes(currentPage)
                          ) {
                            setCurrentPage(jumpTarget)
                          }
                        }}
                      >
                        <Space direction="vertical" size={2} style={{ width: '100%' }}>
                          <Typography.Text strong>{buildTocSequenceTitle(sequence)}</Typography.Text>
                          <Space size={[8, 8]} wrap>
                            <Tag color="cyan">{sequence.toc_sequence_id}</Tag>
                            {isSelected ? <Tag color="magenta">已选中</Tag> : null}
                            {includesCurrentPage ? <Tag color="processing">当前页所在序列</Tag> : null}
                            {pageLabel ? <Tag color="blue">{pageLabel}</Tag> : null}
                            {sequence.entry_count ? <Tag>{sequence.entry_count} 项</Tag> : null}
                            {sequence.root_entry_count ? (
                              <Tag color="geekblue">{sequence.root_entry_count} 根节点</Tag>
                            ) : null}
                          </Space>
                          {buildTocSequenceSummary(sequence, currentPage) ? (
                            <Typography.Text type="secondary">
                              {buildTocSequenceSummary(sequence, currentPage)}
                            </Typography.Text>
                          ) : null}
                          {detail ? (
                            <Typography.Text type="secondary" className="structural-summary-detail">
                              {detail}
                            </Typography.Text>
                          ) : null}
                          {!includesCurrentPage && typeof jumpTarget === 'number' ? (
                            <Typography.Text type="secondary" className="structural-summary-note">
                              点击将跳转到该 TOC 序列的第一页（Page {jumpTarget}）。
                            </Typography.Text>
                          ) : null}
                        </Space>
                      </List.Item>
                    )
                  }}
                />
              ) : null}
              {activePageTables.length > 0 ? (
                <List
                  size="small"
                  bordered
                  header="当前页表格摘要"
                  dataSource={activePageTables}
                  renderItem={(table) => (
                    <List.Item
                      className={
                        table.table_id === selectedStructuralId
                          ? 'structural-summary-item structural-summary-item-selected'
                          : 'structural-summary-item'
                      }
                      onClick={() => setSelectedStructuralId(table.table_id)}
                    >
                      <Space direction="vertical" size={2} style={{ width: '100%' }}>
                        <Typography.Text strong>{table.title || table.table_id}</Typography.Text>
                        <Space size={[8, 8]} wrap>
                          <Tag color="purple">{table.table_id}</Tag>
                          {table.table_id === selectedStructuralId ? (
                            <Tag color="magenta">已选中</Tag>
                          ) : null}
                          {table.detection_method ? (
                            <Tag color="blue">{table.detection_method}</Tag>
                          ) : null}
                          {table.col_count ? <Tag>{table.col_count} 列</Tag> : null}
                          {getSemanticRowCount(table) ? (
                            <Tag color="green">语义 {getSemanticRowCount(table)} 行</Tag>
                          ) : null}
                          {getRawRowCount(table) &&
                          getRawRowCount(table)! > (getSemanticRowCount(table) ?? 0) ? (
                            <Tag color="gold">原始 {getRawRowCount(table)} 行</Tag>
                          ) : null}
                          {table.semantic_compaction?.applied ? (
                            <Tag color="cyan">语义压缩</Tag>
                          ) : null}
                        </Space>
                        {buildTableSummary(table) ? (
                          <Typography.Text type="secondary">
                            {buildTableSummary(table)}
                          </Typography.Text>
                        ) : null}
                        {buildTableCompactionDetail(table) ? (
                          <Typography.Text type="secondary" className="structural-summary-detail">
                            {buildTableCompactionDetail(table)}
                          </Typography.Text>
                        ) : null}
                        {table.semantic_compaction?.applied ? (
                          <Typography.Text type="secondary" className="structural-summary-note">
                            当前工作台优先展示可消费的语义行数，原始审计行数和结构证据未丢失。
                          </Typography.Text>
                        ) : null}
                      </Space>
                    </List.Item>
                  )}
                />
              ) : null}
              {activePageImages.length > 0 ? (
                <List
                  size="small"
                  bordered
                  header="当前页图片 / 图示摘要"
                  dataSource={activePageImages}
                  renderItem={(image) => (
                    <List.Item
                      className={
                        image.image_id === selectedStructuralId
                          ? 'structural-summary-item structural-summary-item-selected'
                          : 'structural-summary-item'
                      }
                      onClick={() => setSelectedStructuralId(image.image_id)}
                    >
                      <Space direction="vertical" size={2} style={{ width: '100%' }}>
                        <Typography.Text strong>{buildImageTitle(image)}</Typography.Text>
                        <Space size={[8, 8]} wrap>
                          <Tag color="gold">{image.image_id}</Tag>
                          {image.image_id === selectedStructuralId ? (
                            <Tag color="magenta">已选中</Tag>
                          ) : null}
                          {image.figure_ref ? <Tag color="blue">{image.figure_ref}</Tag> : null}
                          {image.image_kind_guess ? (
                            <Tag>{image.image_kind_guess}</Tag>
                          ) : null}
                        </Space>
                        {buildImageSummary(image) ? (
                          <Typography.Text type="secondary">
                            {buildImageSummary(image)}
                          </Typography.Text>
                        ) : null}
                      </Space>
                    </List.Item>
                  )}
                />
              ) : null}
              {activePageAlgorithms.length > 0 ? (
                <List
                  size="small"
                  bordered
                  header="当前页算法摘要"
                  dataSource={activePageAlgorithms}
                  renderItem={(algorithm) => (
                    <List.Item
                      className={
                        algorithm.algorithm_id === selectedStructuralId
                          ? 'structural-summary-item structural-summary-item-selected'
                          : 'structural-summary-item'
                      }
                      onClick={() => setSelectedStructuralId(algorithm.algorithm_id)}
                    >
                      <Space direction="vertical" size={2} style={{ width: '100%' }}>
                        <Typography.Text strong>{buildAlgorithmTitle(algorithm)}</Typography.Text>
                        <Space size={[8, 8]} wrap>
                          <Tag color="orange">{algorithm.algorithm_id}</Tag>
                          {algorithm.algorithm_id === selectedStructuralId ? (
                            <Tag color="magenta">已选中</Tag>
                          ) : null}
                          {algorithm.algorithm_ref ? (
                            <Tag color="blue">{algorithm.algorithm_ref}</Tag>
                          ) : null}
                          {algorithm.line_count ? <Tag>{algorithm.line_count} 行</Tag> : null}
                          {algorithm.continued_from_previous_page ? (
                            <Tag color="geekblue">上页续接</Tag>
                          ) : null}
                          {algorithm.continues_to_next_page ? (
                            <Tag color="gold">续至下页</Tag>
                          ) : null}
                        </Space>
                        {buildAlgorithmSummary(algorithm) ? (
                          <Typography.Text type="secondary">
                            {buildAlgorithmSummary(algorithm)}
                          </Typography.Text>
                        ) : null}
                      </Space>
                    </List.Item>
                  )}
                />
              ) : null}
              {activePageEquations.length > 0 ? (
                <List
                  size="small"
                  bordered
                  header="当前页公式摘要"
                  dataSource={activePageEquations}
                  renderItem={(equation) => (
                    <List.Item
                      className={
                        equation.equation_id === selectedStructuralId
                          ? 'structural-summary-item structural-summary-item-selected'
                          : 'structural-summary-item'
                      }
                      onClick={() => setSelectedStructuralId(equation.equation_id)}
                    >
                      <Space direction="vertical" size={2} style={{ width: '100%' }}>
                        <Typography.Text strong>{buildEquationTitle(equation)}</Typography.Text>
                        <Space size={[8, 8]} wrap>
                          <Tag color="lime">{equation.equation_id}</Tag>
                          {equation.equation_id === selectedStructuralId ? (
                            <Tag color="magenta">宸查€変腑</Tag>
                          ) : null}
                          {equation.equation_label ? (
                            <Tag color="blue">{equation.equation_label}</Tag>
                          ) : null}
                        </Space>
                        {buildEquationSummary(equation) ? (
                          <Typography.Text type="secondary">
                            {buildEquationSummary(equation)}
                          </Typography.Text>
                        ) : null}
                      </Space>
                    </List.Item>
                  )}
                />
              ) : null}
              {activePageTocBlocks.length > 0 ? (
                <List
                  size="small"
                  bordered
                  header="当前页 TOC 摘要"
                  dataSource={activePageTocBlocks}
                  renderItem={(toc) => (
                    <List.Item
                      className={
                        toc.toc_id === selectedStructuralId
                          ? 'structural-summary-item structural-summary-item-selected'
                          : 'structural-summary-item'
                      }
                      onClick={() => {
                        setSelectedStructuralId(toc.toc_id)
                        if (toc.toc_sequence_id) {
                          setSelectedTocSequenceId(toc.toc_sequence_id)
                        }
                      }}
                    >
                      <Space direction="vertical" size={2} style={{ width: '100%' }}>
                        <Typography.Text strong>{buildTocTitle(toc)}</Typography.Text>
                        <Space size={[8, 8]} wrap>
                          <Tag color="cyan">{toc.toc_id}</Tag>
                          {toc.toc_id === selectedStructuralId ? (
                            <Tag color="magenta">已选中</Tag>
                          ) : null}
                          {toc.entry_count ? (
                            <Tag color="blue">{toc.entry_count} 项</Tag>
                          ) : null}
                          {toc.toc_sequence_page_index && toc.toc_sequence_length ? (
                            <Tag>
                              序列 {toc.toc_sequence_page_index}/{toc.toc_sequence_length}
                            </Tag>
                          ) : null}
                          {toc.title_inferred ? <Tag color="geekblue">标题推断</Tag> : null}
                          {toc.wrapped_entry_count ? (
                            <Tag color="purple">跨行 {toc.wrapped_entry_count}</Tag>
                          ) : null}
                          {toc.review_required ? <Tag color="error">需复核</Tag> : null}
                        </Space>
                        {buildTocSummary(toc) ? (
                          <Typography.Text type="secondary">
                            {buildTocSummary(toc)}
                          </Typography.Text>
                        ) : null}
                        {buildTocBlockSequenceHint(toc, tocSequenceLookup) ? (
                          <Typography.Text type="secondary" className="structural-summary-detail">
                            {buildTocBlockSequenceHint(toc, tocSequenceLookup)}
                          </Typography.Text>
                        ) : null}
                      </Space>
                    </List.Item>
                  )}
                />
              ) : null}
              <List
                size="small"
                bordered
                header={`当前页 Bounding Box 列表（全部结构框 + 前 ${TEXT_BOUNDING_BOX_LIST_LIMIT} 个文本框）`}
                dataSource={activeBoundingBoxListItems}
                renderItem={(item) => {
                  const kind = getBoundingBoxKind(item)
                  const structuralLabel = activePageStructuralLookup.get(item.id)
                  const displayText = structuralLabel ?? item.text
                  return (
                    <List.Item>
                      <Space size={8} style={{ width: '100%' }}>
                        <Tag color={getBoundingBoxKindTagColor(kind)}>
                          {getBoundingBoxKindLabel(kind)}
                        </Tag>
                        <Typography.Text ellipsis={{ tooltip: displayText }} style={{ width: '100%' }}>
                          {displayText}
                        </Typography.Text>
                      </Space>
                    </List.Item>
                  )
                }}
              />
            </Space>
          )}
        </ProCard>

        <ProCard title="结构化 Markdown 视图" colSpan="34%" className="workbench-col">
          <Space direction="vertical" size={10} style={{ width: '100%' }}>
            <Alert
              type="info"
              showIcon
              message="当前视图展示标准化摘要、解析目录结构与文本预览。完整解析内容请下载 Markdown 文件。"
            />
            <Space wrap>
              {workbench?.ind_review_markdown_download_url || workbench?.full_markdown_download_url ? (
                <Button
                  icon={<DownloadOutlined />}
                  href={buildAssetUrl(
                    workbench.ind_review_markdown_download_url ?? workbench.full_markdown_download_url ?? '',
                  )}
                  target="_blank"
                  title="干净审阅版：正文、目录、表格、图片及说明，隐藏底层 OCR/XML 证据文本。"
                >
                  下载审阅 Markdown（干净版）
                </Button>
              ) : null}
              {workbench?.full_markdown_download_url ? (
                <Button
                  icon={<DownloadOutlined />}
                  href={buildAssetUrl(workbench.full_markdown_download_url)}
                  target="_blank"
                  title="完整证据快照：包含证据清单、AST 对象索引、表格/图片/公式证据与底层解析内容。"
                >
                  下载完整证据快照（含证据清单）
                </Button>
              ) : null}
            </Space>
          </Space>
          {workbench?.markdown ? (
            <div className="markdown-panel">
              <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                {workbench.markdown}
              </ReactMarkdown>
            </div>
          ) : (
            <Empty description="等待解析结果" />
          )}
        </ProCard>

        <ProCard title="审阅结论与提示" colSpan="30%" className="workbench-col">
          <div className="rule-check-panel">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                message={workbench?.rule_checks.message ?? '暂无规则检查结果'}
                type={workbench?.rule_checks.enabled ? 'success' : 'info'}
                showIcon
              />
              {showSingleFileSummary ? (
                <div className="single-file-review-panel">
                  <Space direction="vertical" size={8} style={{ width: '100%' }}>
                    <Space wrap>
                      <Typography.Text strong>{singleFileReviewSummary.title}</Typography.Text>
                      {singleFileReviewSummary.filename ? (
                        <Tag color="blue">{singleFileReviewSummary.filename}</Tag>
                      ) : null}
                      {singleFileReviewSummary.tags.map((tag) => (
                        <Tag key={tag.label} color={tag.color}>
                          {tag.label}
                        </Tag>
                      ))}
                    </Space>
                    <Typography.Text className="rule-detail-line">
                      {singleFileReviewSummary.conclusion}
                    </Typography.Text>
                    <div className="single-file-review-module-list">
                      {singleFileReviewSummary.modules.map((module) => (
                        <div key={module.name} className="single-file-review-module">
                          <Space direction="vertical" size={4} style={{ width: '100%' }}>
                            <Space wrap>
                              <Typography.Text strong>{module.name}</Typography.Text>
                              <Tag color={getSingleFileModuleTagColor(module.status)}>
                                {getSingleFileModuleStatusLabel(module.status)}
                              </Tag>
                            </Space>
                            <Typography.Text type="secondary" className="rule-detail-line">
                              {module.summary}
                            </Typography.Text>
                          </Space>
                        </div>
                      ))}
                    </div>
                  </Space>
                </div>
              ) : null}
              {showDemoPanels && workbench?.demo_summary ? (
                <div className="demo-summary-panel">
                  <Space direction="vertical" size={8} style={{ width: '100%' }}>
                    <Space wrap>
                      <Typography.Text strong>演示概览</Typography.Text>
                      <Tag color="blue">{getReviewProjectionStatusLabel(workbench.demo_summary.phase_status)}</Tag>
                      <Tag color="green">
                        法规来源 {workbench.demo_summary.summary.closed_source_count}/
                        {workbench.demo_summary.summary.source_count}
                      </Tag>
                      <Tag>资料要求 {workbench.demo_summary.summary.dossier_requirement_count}</Tag>
                      <Tag color="gold">
                        前置条件 {workbench.demo_summary.summary.missing_prerequisite_fact_keys.length}
                      </Tag>
                      <Tag color="default">
                        硬判定 {workbench.demo_summary.summary.deterministic_dossier_decision_count}
                      </Tag>
                    </Space>
                    <Typography.Text className="rule-detail-line">
                      {workbench.demo_summary.customer_demo_message}
                    </Typography.Text>
                    <Typography.Text type="secondary" className="rule-detail-line">
                      {workbench.demo_summary.evidence_boundary}
                    </Typography.Text>
                    <Typography.Text type="secondary" className="rule-detail-line">
                      下一步：{workbench.demo_summary.recommended_next_action_label}
                    </Typography.Text>
                    {workbench.demo_report_markdown_download_url ? (
                      <Button
                        icon={<DownloadOutlined />}
                        href={buildAssetUrl(workbench.demo_report_markdown_download_url)}
                        size="small"
                      >
                        下载演示报告
                      </Button>
                    ) : null}
                  </Space>
                </div>
              ) : null}
              {showDemoPanels && workbench?.demo_run ? (
                <div className="demo-run-panel">
                  <Space direction="vertical" size={8} style={{ width: '100%' }}>
                    <Space wrap>
                      <Typography.Text strong>演示检查清单</Typography.Text>
                      <Tag color="blue">{getReviewProjectionStatusLabel(workbench.demo_run.run_mode)}</Tag>
                      <Tag color={workbench.demo_run.run_status === 'ready' ? 'green' : workbench.demo_run.run_status === 'asset_missing' ? 'red' : 'orange'}>
                        {getReviewProjectionStatusLabel(workbench.demo_run.run_status)}
                      </Tag>
                      <Tag color="green">
                        已就绪 {workbench.demo_run.summary.ready_step_count}/{workbench.demo_run.summary.run_step_count}
                      </Tag>
                      <Tag color="gold">
                        前置条件 {workbench.demo_run.summary.prerequisite_prompt_step_count}
                      </Tag>
                      <Tag color="orange">
                        待复核 {workbench.demo_run.summary.review_required_step_count}
                      </Tag>
                    </Space>
                    <Typography.Text type="secondary" className="rule-detail-line">
                      {workbench.demo_run.evidence_boundary}
                    </Typography.Text>
                    <div className="demo-run-step-list">
                      {workbench.demo_run.run_steps.map((step, index) => (
                        <div key={step.step_id} className="demo-run-step">
                          <Space direction="vertical" size={4} style={{ width: '100%' }}>
                            <Space wrap>
                              <Tag>{index + 1}</Tag>
                              <Typography.Text strong>{step.title}</Typography.Text>
                              <Tag color={step.status === 'review_required' ? 'orange' : step.status === 'prerequisite_prompt' ? 'gold' : step.status === 'not_available' ? 'red' : 'green'}>
                                {getReviewProjectionStatusLabel(step.status)}
                              </Tag>
                            </Space>
                            <Typography.Text className="rule-detail-line">{step.talk_track}</Typography.Text>
                            {step.focus_items.length > 0 ? (
                              <div className="demo-run-focus-list">
                                {step.focus_items.map((item) => (
                                  <Tag key={`${step.step_id}-${item}`} color="cyan">
                                    {item}
                                  </Tag>
                                ))}
                              </div>
                            ) : null}
                          </Space>
                        </div>
                      ))}
                    </div>
                    <Space wrap>
                      {workbench.demo_run.asset_urls.demo_report_markdown ? (
                        <Button
                          icon={<DownloadOutlined />}
                          href={buildAssetUrl(workbench.demo_run.asset_urls.demo_report_markdown)}
                          size="small"
                        >
                          打开演示报告
                        </Button>
                      ) : null}
                      {workbench.demo_run.asset_urls.demo_script_markdown ? (
                        <Button
                          icon={<DownloadOutlined />}
                          href={buildAssetUrl(workbench.demo_run.asset_urls.demo_script_markdown)}
                          size="small"
                        >
                          打开演示脚本
                        </Button>
                      ) : null}
                    </Space>
                  </Space>
                </div>
              ) : null}
              {showDemoPanels && workbench?.demo_scenario ? (
                <div className="demo-scenario-panel">
                  <Space direction="vertical" size={8} style={{ width: '100%' }}>
                    <Space wrap>
                      <Typography.Text strong>演示场景</Typography.Text>
                      <Tag color="blue">{getReviewProjectionStatusLabel(workbench.demo_scenario.scenario_status)}</Tag>
                      <Tag color="cyan">{getReviewProjectionStatusLabel(workbench.demo_scenario.recommended_upload_mode)}</Tag>
                      <Tag color="green">
                        步骤 {workbench.demo_scenario.summary.walkthrough_step_count}
                      </Tag>
                      <Tag color="gold">
                        待补充 {workbench.demo_scenario.summary.expected_missing_prerequisite_count}
                      </Tag>
                      <Tag color={workbench.demo_scenario.summary.expected_issue_count > 0 ? 'orange' : 'green'}>
                        复核项 {workbench.demo_scenario.summary.expected_issue_count}
                      </Tag>
                    </Space>
                    <Typography.Text className="rule-detail-line">{workbench.demo_scenario.demo_goal}</Typography.Text>
                    <Typography.Text type="secondary" className="rule-detail-line">
                      {workbench.demo_scenario.evidence_boundary}
                    </Typography.Text>
                    <Typography.Text strong className="rule-detail-line">
                      样本：{getReviewProjectionStatusLabel(workbench.demo_scenario.recommended_sample_profile.sample_kind)}
                    </Typography.Text>
                    <Space wrap size={[6, 4]}>
                      <Tag color="cyan">
                        本地证据 {workbench.demo_scenario.recommended_sample_profile.must_have_local_evidence.length}
                      </Tag>
                      <Tag color="gold">
                        待补充条件 {workbench.demo_scenario.recommended_sample_profile.intentionally_missing_prerequisites.length}
                      </Tag>
                      <Tag color="green">演示达成项 {workbench.demo_scenario.demo_success_criteria.length}</Tag>
                      <Tag color="red">不做结论 {workbench.demo_scenario.do_not_claim.length}</Tag>
                    </Space>
                    {workbench.demo_scenario.primary_asset_urls.demo_report_markdown ? (
                      <Button
                        icon={<DownloadOutlined />}
                        href={buildAssetUrl(workbench.demo_scenario.primary_asset_urls.demo_report_markdown)}
                        size="small"
                      >
                        打开场景报告
                      </Button>
                    ) : null}
                    {workbench.demo_script_markdown_download_url ? (
                      <Button
                        icon={<DownloadOutlined />}
                        href={buildAssetUrl(workbench.demo_script_markdown_download_url)}
                        size="small"
                      >
                        打开演示脚本
                      </Button>
                    ) : null}
                  </Space>
                </div>
              ) : null}
              {showDemoPanels && workbench?.demo_flow ? (
                <div className="demo-flow-panel">
                  <Space direction="vertical" size={8} style={{ width: '100%' }}>
                    <Space wrap>
                      <Typography.Text strong>客户演示路径</Typography.Text>
                      <Tag color="blue">{getReviewProjectionStatusLabel(workbench.demo_flow.summary.recommended_demo_mode)}</Tag>
                      <Tag color="green">
                        步骤 {workbench.demo_flow.summary.walkthrough_step_count}
                      </Tag>
                      <Tag color="gold">
                        前置条件 {workbench.demo_flow.summary.missing_prerequisite_count}
                      </Tag>
                      <Tag color={workbench.demo_flow.summary.issue_count > 0 ? 'orange' : 'green'}>
                        复核项 {workbench.demo_flow.summary.issue_count}
                      </Tag>
                    </Space>
                    <Typography.Text type="secondary" className="rule-detail-line">
                      {workbench.demo_flow.evidence_boundary}
                    </Typography.Text>
                    <div className="demo-flow-step-list">
                      {workbench.demo_flow.walkthrough_steps.map((step, index) => (
                        <div key={step.step_id} className="demo-flow-step">
                          <Space direction="vertical" size={4} style={{ width: '100%' }}>
                            <Space wrap>
                              <Tag>{index + 1}</Tag>
                              <Typography.Text strong>{step.title}</Typography.Text>
                              <Tag color={step.status === 'review_required' ? 'orange' : step.status === 'prerequisite_prompt' ? 'gold' : 'green'}>
                                {getReviewProjectionStatusLabel(step.status)}
                              </Tag>
                            </Space>
                            <Typography.Text className="rule-detail-line">{step.talk_track}</Typography.Text>
                            {step.focus_items.length > 0 ? (
                              <div className="demo-flow-focus-list">
                                {step.focus_items.map((item) => (
                                  <Tag key={`${step.step_id}-${item}`} color="cyan">
                                    {item}
                                  </Tag>
                                ))}
                              </div>
                            ) : null}
                            {step.asset_url ? (
                              <Button icon={<DownloadOutlined />} href={buildAssetUrl(step.asset_url)} size="small">
                                打开材料
                              </Button>
                            ) : null}
                          </Space>
                        </div>
                      ))}
                    </div>
                  </Space>
                </div>
              ) : null}
              {showProjectReadinessPanels && workbench?.regulatory_readiness ? (
                <div className="regulatory-readiness-panel">
                  <Space direction="vertical" size={8} style={{ width: '100%' }}>
                    <Space wrap>
                      <Typography.Text strong>法规来源就绪度</Typography.Text>
                      <Tag color="blue">{getReviewProjectionStatusLabel(workbench.regulatory_readiness.phase)}</Tag>
                      <Tag color="cyan">{getReviewProjectionStatusLabel(workbench.regulatory_readiness.phase_status)}</Tag>
                      <Tag color="green">
                        闭合 {workbench.regulatory_readiness.summary.closed_source_count}/
                        {workbench.regulatory_readiness.summary.source_count}
                      </Tag>
                    </Space>
                    <Typography.Text type="secondary" className="rule-detail-line">
                      {workbench.regulatory_readiness.summary.demo_value_statement}
                    </Typography.Text>
                    <div className="regulatory-readiness-triage-list">
                      {workbench.regulatory_readiness.rule_triage_categories.map((category) => {
                        const firstMatchingRow = regulatoryReadinessRows.find((row) => row.triage === category)
                        return (
                          <Tag
                            key={`readiness-triage-${category}`}
                            color={firstMatchingRow?.triageColor ?? 'default'}
                          >
                            {firstMatchingRow?.triageLabel ?? category}
                          </Tag>
                        )
                      })}
                    </div>
                    <div className="regulatory-readiness-matrix">
                      {regulatoryReadinessRows.map((row) => (
                        <div key={row.key} className="regulatory-readiness-row">
                          <Space direction="vertical" size={5} style={{ width: '100%' }}>
                            <Space wrap>
                              <Tag color={row.statusColor}>{row.coverageLabel}</Tag>
                              <Tag color={row.triageColor}>{row.triageLabel}</Tag>
                              <Typography.Text strong className="regulatory-readiness-source-title">
                                {row.sourceFile}
                              </Typography.Text>
                            </Space>
                            <Typography.Text type="secondary" className="rule-detail-line">
                              {row.productRoleLabel} | {row.regulationId}
                            </Typography.Text>
                            <div className="regulatory-readiness-metrics">
                              <Tag>{row.clauseSummary}</Tag>
                              <Tag>{row.ruleSummary}</Tag>
                            </div>
                            <Typography.Text className="rule-detail-line">
                              {row.automationBoundary}
                            </Typography.Text>
                          </Space>
                        </div>
                      ))}
                    </div>
                  </Space>
                </div>
              ) : null}
              {showProjectReadinessPanels && workbench?.dossier_checklist ? (
                <div className="dossier-checklist-panel">
                  <Space direction="vertical" size={8} style={{ width: '100%' }}>
                    <Space wrap>
                      <Typography.Text strong>资料清单适用性</Typography.Text>
                      <Tag color="gold">{getReviewProjectionStatusLabel(workbench.dossier_checklist.applicability_mode)}</Tag>
                      <Tag>要求 {workbench.dossier_checklist.summary.requirement_count}</Tag>
                      <Tag color="gold">
                        需前置条件 {workbench.dossier_checklist.summary.prerequisite_required_count}
                      </Tag>
                      <Tag>硬判定 {workbench.dossier_checklist.summary.deterministic_decision_count}</Tag>
                    </Space>
                    <Typography.Text type="secondary" className="rule-detail-line">
                      {workbench.dossier_checklist.summary.evidence_boundary}
                    </Typography.Text>
                    <div className="dossier-prerequisite-list">
                      {workbench.dossier_checklist.prerequisite_facts.map((fact) => (
                        <div key={`dossier-prerequisite-${fact.fact_key}`} className="dossier-prerequisite-item">
                          <Space direction="vertical" size={4} style={{ width: '100%' }}>
                            <Space wrap size={[6, 4]}>
                              <Tag color={fact.status === 'present' ? 'green' : 'gold'}>
                                {getReviewProjectionStatusLabel(fact.status)}
                              </Tag>
                              <Typography.Text strong>{fact.label}</Typography.Text>
                              <Tag color={fact.confidence === 'high' ? 'green' : fact.confidence === 'none' ? 'default' : 'blue'}>
                                {getReviewProjectionStatusLabel(fact.confidence)}
                              </Tag>
                            </Space>
                            <Typography.Text className="rule-detail-line">
                              {fact.status === 'present' ? String(fact.value ?? '') : '待补充'}
                            </Typography.Text>
                            <Space wrap size={[6, 4]}>
                              <Tag>{fact.source_label ?? fact.source ?? '暂无来源'}</Tag>
                              <Tag color={fact.evidence_status === 'local_evidence_present' ? 'cyan' : 'gold'}>
                                {getReviewProjectionStatusLabel(fact.evidence_status ?? fact.status)}
                              </Tag>
                            </Space>
                            <Typography.Text type="secondary" className="rule-detail-line">
                              {fact.review_action ?? fact.description}
                            </Typography.Text>
                          </Space>
                        </div>
                      ))}
                    </div>
                    <div className="dossier-checklist-list">
                      {workbench.dossier_checklist.items.slice(0, 6).map((item) => (
                        <div key={item.requirement_id} className="dossier-checklist-item">
                          <Space direction="vertical" size={5} style={{ width: '100%' }}>
                            <Space wrap>
                              <Tag color={item.applicability_status === 'prerequisite_required' ? 'gold' : 'orange'}>
                                {getReviewProjectionStatusLabel(item.applicability_status)}
                              </Tag>
                              <Tag>条款 {item.source_article_no}</Tag>
                              <Tag>{item.applicable_stage}</Tag>
                              <Tag>{item.requirement_type}</Tag>
                            </Space>
                            <Typography.Text strong className="rule-detail-line">
                              {item.requirement_id}
                            </Typography.Text>
                            {item.registration_classes.length > 0 ? (
                              <div className="dossier-checklist-chip-list">
                                {item.registration_classes.map((registrationClass) => (
                                  <Tag key={`${item.requirement_id}-${registrationClass}`} color="blue">
                                    {registrationClass}
                                  </Tag>
                                ))}
                              </div>
                            ) : null}
                            {item.expected_material_evidence.length > 0 ? (
                              <div className="dossier-checklist-chip-list">
                                {item.expected_material_evidence.map((evidence) => (
                                  <Tag key={`${item.requirement_id}-${evidence}`}>{evidence}</Tag>
                                ))}
                              </div>
                            ) : null}
                            {item.blocking_prerequisite_fact_keys.length > 0 ? (
                              <Typography.Text type="secondary" className="rule-detail-line">
                                前置条件: {item.blocking_prerequisite_fact_keys.join(' / ')}
                              </Typography.Text>
                            ) : null}
                            <Typography.Text className="rule-detail-line">
                              {item.review_focus || item.requirement_text_preview}
                            </Typography.Text>
                            <Typography.Text type="secondary" className="rule-detail-line">
                              {item.automation_boundary}
                            </Typography.Text>
                          </Space>
                        </div>
                      ))}
                    </div>
                    {workbench.dossier_checklist.items.length > 6 ? (
                      <Typography.Text type="secondary" className="rule-detail-line">
                        当前预览 6/{workbench.dossier_checklist.items.length} 条；完整清单保留在 workbench payload 中。
                      </Typography.Text>
                    ) : null}
                  </Space>
                </div>
              ) : null}
              {showProjectReadinessPanels && workbench?.content_consistency ? (
                <div className="content-consistency-panel">
                  <Space direction="vertical" size={8} style={{ width: '100%' }}>
                    <Space wrap>
                      <Typography.Text strong>内容一致性</Typography.Text>
                      <Tag color="blue">检查项 {workbench.content_consistency.summary.check_count}</Tag>
                      <Tag color={workbench.content_consistency.summary.issue_count > 0 ? 'orange' : 'green'}>
                        复核项 {workbench.content_consistency.summary.issue_count}
                      </Tag>
                      <Tag color="default">
                        硬判定 {workbench.content_consistency.summary.deterministic_rule_verdict_count}
                      </Tag>
                    </Space>
                    <Typography.Text type="secondary" className="rule-detail-line">
                      {workbench.content_consistency.evidence_boundary}
                    </Typography.Text>
                    <div className="content-consistency-list">
                      {workbench.content_consistency.checks.map((check) => (
                        <div key={check.check_id} className="content-consistency-item">
                          <Space direction="vertical" size={5} style={{ width: '100%' }}>
                            <Space wrap>
                              <Tag color={check.status === 'review_required' ? 'orange' : check.status === 'consistent' ? 'green' : 'gold'}>
                                {getReviewProjectionStatusLabel(check.status)}
                              </Tag>
                              <Tag>{check.field_name}</Tag>
                              <Tag>可比资料 {check.comparable_package_count}</Tag>
                              <Tag>复核项 {check.issue_count}</Tag>
                            </Space>
                            <Typography.Text strong className="rule-detail-line">
                              {check.title}
                            </Typography.Text>
                            <Typography.Text type="secondary" className="rule-detail-line">
                              {check.description}
                            </Typography.Text>
                            {check.issues.map((issue) => (
                              <div key={`${check.check_id}-${issue.sequence_package_id}-${issue.field_name}`} className="content-consistency-issue">
                                <Typography.Text className="rule-detail-line">
                                  {issue.message}
                                </Typography.Text>
                                <Space wrap>
                                  <Tag color="gold">{issue.sequence_package_id}</Tag>
                                  <Tag>{issue.expected_source}: {issue.expected_value}</Tag>
                                  <Tag>{issue.observed_source}: {issue.observed_value}</Tag>
                                </Space>
                                <Typography.Text type="secondary" className="rule-detail-line">
                                  {issue.review_recommendation}
                                </Typography.Text>
                              </div>
                            ))}
                            <Typography.Text type="secondary" className="rule-detail-line">
                              {check.automation_boundary}
                            </Typography.Text>
                          </Space>
                        </div>
                      ))}
                    </div>
                  </Space>
                </div>
              ) : null}
              {workbench?.rule_checks.summary ? (
                <div className="rule-summary-chip-list">
                  {ruleCheckStatusFilterOptions.map((option) => (
                    <Tag
                      key={option.key}
                      color={ruleCheckStatusFilter === option.key ? 'magenta' : option.color}
                      className="clickable-tag"
                      onClick={() => handleRuleSummaryFilterClick(option.key)}
                    >
                      {option.label}: {option.count}
                    </Tag>
                  ))}
                </div>
              ) : null}
              {showProjectReadinessPanels && workbench?.rule_checks.summary?.submission_scope_overview ? (
                <div className="rule-group-summary-card">
                  <Space direction="vertical" size={6} style={{ width: '100%' }}>
                    <Space wrap>
                      <Tag color="cyan">
                        上传判定: {workbench.rule_checks.summary.submission_scope_overview.upload_mode_label}
                      </Tag>
                      <Tag color="blue">
                        可执行作用域: {workbench.rule_checks.summary.submission_scope_overview.available_scope_labels.join(' / ')}
                      </Tag>
                      {(workbench.rule_checks.summary.submission_scope_overview.blocked_scope_labels ?? []).length > 0 ? (
                        <Tag color="gold">
                          暂不适用: {workbench.rule_checks.summary.submission_scope_overview.blocked_scope_labels.join(' / ')}
                        </Tag>
                      ) : null}
                      <Tag>序列包 {workbench.rule_checks.summary.submission_scope_overview.sequence_package_count}</Tag>
                      <Tag>注册行为 {workbench.rule_checks.summary.submission_scope_overview.regulatory_activity_count}</Tag>
                      <Tag>申请项目 {workbench.rule_checks.summary.submission_scope_overview.application_project_count}</Tag>
                    </Space>
                    <Typography.Text type="secondary" className="rule-detail-line">
                      {workbench.rule_checks.summary.submission_scope_overview.scope_notice}
                    </Typography.Text>
                  </Space>
                </div>
              ) : null}
              {showProjectReadinessPanels && workbench?.rule_checks.summary?.scope_transition_overview ? (
                <div className="rule-group-summary-card">
                  <Space direction="vertical" size={6} style={{ width: '100%' }}>
                    <Space wrap>
                      <Tag color="purple">
                        作用域变化: {workbench.rule_checks.summary.scope_transition_overview.transition_status_label}
                      </Tag>
                      <Tag color="geekblue">
                        主因: {workbench.rule_checks.summary.scope_transition_overview.primary_reason_label}
                      </Tag>
                      <Tag>
                        上传预判: {workbench.rule_checks.summary.scope_transition_overview.upload_mode_label}
                      </Tag>
                      <Tag>
                        最终判定: {workbench.rule_checks.summary.scope_transition_overview.final_mode_label}
                      </Tag>
                      {(workbench.rule_checks.summary.scope_transition_overview.added_scope_labels ?? []).length > 0 ? (
                        <Tag color="green">
                          新增: {workbench.rule_checks.summary.scope_transition_overview.added_scope_labels.join(' / ')}
                        </Tag>
                      ) : null}
                      {(workbench.rule_checks.summary.scope_transition_overview.removed_scope_labels ?? []).length > 0 ? (
                        <Tag color="volcano">
                          收缩: {workbench.rule_checks.summary.scope_transition_overview.removed_scope_labels.join(' / ')}
                        </Tag>
                      ) : null}
                    </Space>
                    <Typography.Text type="secondary" className="rule-detail-line">
                      {workbench.rule_checks.summary.scope_transition_overview.transition_notice}
                    </Typography.Text>
                    <Typography.Text type="secondary" className="rule-detail-line">
                      Action: {workbench.rule_checks.summary.scope_transition_overview.recommended_action_label}
                    </Typography.Text>
                    {workbench.rule_checks.summary.scope_transition_overview.guidance_title ||
                    (workbench.rule_checks.summary.scope_transition_overview.guidance_steps ?? []).length > 0 ? (
                      <div className="rule-detail-block">
                        <Space direction="vertical" size={6} style={{ width: '100%' }}>
                          <Space wrap>
                            <Typography.Text strong>整改建议</Typography.Text>
                            <Tag
                              color={getGuidancePriorityColor(
                                workbench.rule_checks.summary.scope_transition_overview.guidance_priority,
                              )}
                            >
                              {getGuidancePriorityLabel(
                                workbench.rule_checks.summary.scope_transition_overview.guidance_priority,
                              )}
                            </Tag>
                          </Space>
                          {workbench.rule_checks.summary.scope_transition_overview.guidance_title ? (
                            <Typography.Text>
                              {workbench.rule_checks.summary.scope_transition_overview.guidance_title}
                            </Typography.Text>
                          ) : null}
                          {workbench.rule_checks.summary.scope_transition_overview.guidance_summary ? (
                            <Typography.Text type="secondary" className="rule-detail-line">
                              {workbench.rule_checks.summary.scope_transition_overview.guidance_summary}
                            </Typography.Text>
                          ) : null}
                          {(() => {
                            const guidanceTargetDetails =
                              workbench.rule_checks.summary.scope_transition_overview.guidance_target_details ?? []
                            if (guidanceTargetDetails.length === 0) {
                              return (workbench.rule_checks.summary.scope_transition_overview.guidance_targets ?? [])
                                .length > 0 ? (
                                <div className="rule-detail-chip-list">
                                  {(workbench.rule_checks.summary.scope_transition_overview.guidance_targets ?? []).map(
                                    (target) => (
                                      <Tag key={`guidance-target-${target}`} color="geekblue">
                                        {target}
                                      </Tag>
                                    ),
                                  )}
                                </div>
                              ) : null
                            }

                            const groupedTargetDetails = guidanceTargetDetails.reduce<
                              Array<{ targetType: string; items: typeof guidanceTargetDetails }>
                            >((groups, detail) => {
                              const existingGroup = groups.find((group) => group.targetType === detail.target_type)
                              if (existingGroup) {
                                existingGroup.items.push(detail)
                                return groups
                              }
                              groups.push({ targetType: detail.target_type, items: [detail] })
                              return groups
                            }, [])

                            return (
                              <Space direction="vertical" size={4} style={{ width: '100%' }}>
                                {groupedTargetDetails.map((group) => (
                                  <div key={`guidance-group-${group.targetType}`}>
                                    <Typography.Text type="secondary" className="rule-detail-line">
                                      {getGuidanceTargetTypeLabel(group.targetType)}
                                    </Typography.Text>
                                    <div className="rule-detail-chip-list">
                                      {group.items.map((detail) => {
                                        const focusRuleId =
                                          detail.target_type === 'rule_scope'
                                            ? resolveGuidanceRuleGroupFocusTarget({
                                                guidanceRuleGroups: [detail.label],
                                                ruleItems: workbench?.rule_checks.items ?? [],
                                              })
                                            : null
                                        return (
                                          <Tag
                                            key={`guidance-detail-${group.targetType}-${detail.label}`}
                                            color={detail.target_type === 'rule_scope' ? 'purple' : 'geekblue'}
                                            className={focusRuleId ? 'clickable-tag' : undefined}
                                            onClick={focusRuleId ? () => focusRule(focusRuleId) : undefined}
                                          >
                                            {detail.label}
                                          </Tag>
                                        )
                                      })}
                                    </div>
                                    <Space direction="vertical" size={2} style={{ width: '100%' }}>
                                      {group.items.map((detail) => (
                                        <Typography.Text
                                          key={`guidance-detail-description-${group.targetType}-${detail.label}`}
                                          type="secondary"
                                          className="rule-detail-line"
                                        >
                                          {detail.label}: {detail.description}
                                        </Typography.Text>
                                      ))}
                                    </Space>
                                  </div>
                                ))}
                              </Space>
                            )
                          })()}
                          <Space direction="vertical" size={2} style={{ width: '100%' }}>
                            {(workbench.rule_checks.summary.scope_transition_overview.guidance_steps ?? []).map(
                              (step, index) => (
                                <Typography.Text key={`guidance-step-${index + 1}`} className="rule-detail-line">
                                  {index + 1}. {step}
                                </Typography.Text>
                              ),
                            )}
                          </Space>
                        </Space>
                      </div>
                    ) : null}
                  </Space>
                </div>
              ) : null}
              {ruleGroupSummaries.length > 0 ? (
                <div className="rule-group-summary-list">
                  {ruleGroupSummaries.map((group) => (
                    <div key={group.group_id} className="rule-group-summary-card">
                      <Space direction="vertical" size={6} style={{ width: '100%' }}>
                        <Space wrap>
                          <Typography.Text strong>{group.title}</Typography.Text>
                          <Tag color="blue">规则 {group.total_rules}</Tag>
                          {[
                            { status: 'fail' as RuleCheckStatusFilter, label: '失败', count: group.fail_count },
                            { status: 'warn' as RuleCheckStatusFilter, label: '预警', count: group.warn_count },
                            { status: 'pass' as RuleCheckStatusFilter, label: '通过', count: group.pass_count },
                            { status: 'na' as RuleCheckStatusFilter, label: '不适用', count: group.na_count },
                          ].map((item) => {
                            const isActive =
                              activeRuleGroupStatusFilter?.groupId === group.group_id &&
                              activeRuleGroupStatusFilter.status === item.status
                            return (
                              <Tag
                                key={`${group.group_id}-${item.status}`}
                                color={isActive ? 'magenta' : getRuleStatusColor(item.status)}
                                className={item.count > 0 ? 'clickable-tag' : undefined}
                                onClick={
                                  item.count > 0
                                    ? () => handleRuleGroupStatusFilterClick(group.group_id, item.status)
                                    : undefined
                                }
                              >
                                {item.label} {item.count}
                              </Tag>
                            )
                          })}
                        </Space>
                        <Typography.Text type="secondary" className="rule-detail-line">
                          {group.description}
                        </Typography.Text>
                        {(group.focus_rule_ids ?? []).length > 0 ? (
                          <Space direction="vertical" size={4} style={{ width: '100%' }}>
                            {(group.focus_rule_ids ?? []).map((ruleId) => {
                              const ruleItem = ruleItemsById.get(ruleId)
                              if (!ruleItem) {
                                return null
                              }
                              return (
                                <Button
                                  key={`${group.group_id}-${ruleId}`}
                                  size="small"
                                  type="link"
                                  onClick={() => focusRule(ruleId)}
                                  style={{ paddingInline: 0, width: 'fit-content' }}
                                >
                                  {buildRuleHeaderPresentation(ruleItem.rule_id).title}:{' '}
                                  {getRuleSummary(ruleItem.rule_id, ruleItem.status, ruleItem.message)}
                                </Button>
                              )
                            })}
                          </Space>
                        ) : null}
                      </Space>
                    </div>
                  ))}
                </div>
              ) : null}
              {structureAuditRecords.length > 0 ? (
                <div className="structure-audit-panel">
                  <Space direction="vertical" style={{ width: '100%' }} size={8}>
                    <Space wrap>
                      <Typography.Text strong>结构审计</Typography.Text>
                      <Tag color="blue">当前 {visibleStructureAuditRecordCount}</Tag>
                      <Tag>总记录 {structureAuditRecords.length}</Tag>
                      {focusedRuleId ? <Tag color="magenta">已聚焦规则 {focusedRuleId}</Tag> : null}
                      {workbench?.structure_audit_download_url ? (
                        <Button
                          size="small"
                          icon={<DownloadOutlined />}
                          href={buildAssetUrl(workbench.structure_audit_download_url)}
                          target="_blank"
                        >
                          下载结构审计
                        </Button>
                      ) : null}
                      {workbench?.structure_audit_markdown_download_url ? (
                        <Button
                          size="small"
                          icon={<DownloadOutlined />}
                          href={buildAssetUrl(workbench.structure_audit_markdown_download_url)}
                          target="_blank"
                        >
                          下载结构审计报告
                        </Button>
                      ) : null}
                      {focusedRuleId ? (
                        <Button size="small" onClick={() => setFocusedRuleId(null)}>
                          清除聚焦
                        </Button>
                      ) : null}
                    </Space>
                    <Space wrap className="structure-audit-controls">
                      <Select
                        size="small"
                        value={structureAuditSeverityFilter}
                        style={{ minWidth: 124 }}
                        options={[
                          { value: 'warnings', label: '只看预警' },
                          { value: 'passes', label: '只看通过' },
                          { value: 'all', label: '查看全部' },
                        ]}
                        onChange={(value) =>
                          setStructureAuditSeverityFilter(value as 'warnings' | 'passes' | 'all')
                        }
                      />
                      <Select
                        size="small"
                        value={structureAuditIssueFilter}
                        style={{ minWidth: 188 }}
                        options={structureAuditIssueOptions}
                        onChange={(value) => setStructureAuditIssueFilter(value)}
                      />
                      <Select
                        size="small"
                        value={structureAuditFilenameFilter}
                        style={{ minWidth: 220 }}
                        options={structureAuditFilenameOptions}
                        onChange={(value) => setStructureAuditFilenameFilter(value)}
                      />
                    </Space>
                    {structureAuditIssueSummary.length > 0 ? (
                      <div className="structure-audit-summary-chip-list">
                        {structureAuditIssueSummary.map((item) => (
                          <Tag
                            key={item.issueKey}
                            color={structureAuditIssueFilter === item.issueKey ? 'magenta' : 'orange'}
                            className="clickable-tag"
                            onClick={() =>
                              setStructureAuditIssueFilter(
                                toggleStructureAuditFilterValue(structureAuditIssueFilter, item.issueKey),
                              )
                            }
                          >
                            {item.label}: {item.count}
                          </Tag>
                        ))}
                      </div>
                    ) : null}
                    {renderStructureAuditGroups()}
                  </Space>
                </div>
              ) : null}
              {workbench?.rule_checks.enabled && (workbench.rule_checks.items?.length ?? 0) > 0 ? (
                <div id="rule-check-list" className="rule-check-list-panel">
                  <Space direction="vertical" size={8} style={{ width: '100%' }}>
                    <Space wrap>
                      <Typography.Text strong>规则明细</Typography.Text>
                      <Tag color="blue">当前筛选: {getRuleCheckFilterLabel(ruleCheckStatusFilter)}</Tag>
                      {activeRuleGroupSummary ? <Tag color="purple">模块: {activeRuleGroupSummary.title}</Tag> : null}
                      <Tag>当前 {ruleCheckItems.length}</Tag>
                      <Tag>总规则 {rawRuleCheckItems.length}</Tag>
                      {ruleCheckStatusFilter !== DEFAULT_RULE_CHECK_STATUS_FILTER ? (
                        <Button
                          size="small"
                          onClick={() => {
                            setRuleCheckStatusFilter(DEFAULT_RULE_CHECK_STATUS_FILTER)
                            setActiveRuleGroupStatusFilter(null)
                          }}
                        >
                          查看需处理与通过规则
                        </Button>
                      ) : null}
                      {activeRuleGroupSummary ? (
                        <Button size="small" onClick={clearRuleGroupStatusFilter}>
                          清除模块筛选
                        </Button>
                      ) : null}
                    </Space>
                    {ruleCheckStatusFilter === DEFAULT_RULE_CHECK_STATUS_FILTER ? (
                      <Typography.Text type="secondary" className="rule-detail-line">
                        默认隐藏不适用规则；需要查看时点击上方“不适用”统计。
                      </Typography.Text>
                    ) : null}
                    {ruleCheckItems.length > 0 ? (
                      <List
                        size="small"
                        dataSource={ruleCheckItems}
                        renderItem={(item) => {
                          const ruleHeader = buildRuleHeaderPresentation(item.rule_id)
                          return (
                          <List.Item
                            id={`rule-check-item-${item.rule_id}`}
                            className={
                              item.rule_id === focusedRuleId
                                ? 'rule-check-item rule-check-item-focused'
                                : 'rule-check-item'
                            }
                          >
                      <Space direction="vertical" style={{ width: '100%' }} size={6}>
                        <Space wrap>
                          <Tag color={getRuleStatusColor(item.status)}>{getRuleStatusLabel(item.status)}</Tag>
                          <Tag color={getRuleCategoryColor(item.category)}>{getRuleCategoryLabel(item.category)}</Tag>
                          <Typography.Text strong>{ruleHeader.title}</Typography.Text>
                          {ruleHeader.secondaryRuleId ? (
                            <Typography.Text type="secondary">{ruleHeader.secondaryRuleId}</Typography.Text>
                          ) : null}
                          {item.rule_id === focusedRuleId ? <Tag color="magenta">当前聚焦</Tag> : null}
                        </Space>
                        <Typography.Text>{getRuleSummary(item.rule_id, item.status, item.message)}</Typography.Text>
                        {renderRuleDetails(item, navigationAuditRecords, handleRuleNavigation)}
                      </Space>
                          </List.Item>
                          )
                        }}
                      />
                    ) : (
                      <Empty
                        description={
                          activeRuleGroupSummary
                            ? `当前模块下暂无${getRuleCheckFilterLabel(ruleCheckStatusFilter)}明细`
                            : `当前筛选下暂无${getRuleCheckFilterLabel(ruleCheckStatusFilter)}明细`
                        }
                      />
                    )}
                  </Space>
                </div>
              ) : (
                <Empty description="暂无可展示的规则检查条目" />
              )}
            </Space>
          </div>
        </ProCard>
      </ProCard>
    </ProCard>
  )
}


import { CheckCircleOutlined, ExclamationCircleOutlined, FileSearchOutlined, LinkOutlined } from '@ant-design/icons'
import { Alert, Button, Collapse, Empty, List, Space, Tag, Typography } from 'antd'

import { buildProjectFindingLayers } from '../projectFindingLayers.js'
import type { ProjectFindingLayer } from '../projectFindingLayers.js'
import type { WorkbenchPayload } from '../types'
import { PdfPresentationContract } from './PdfPresentationContract'

interface ProjectFindingLayersPanelProps {
  workbench: WorkbenchPayload
  selectedPath?: string | null
  onPathSelect?: (path: string) => void
  onRuleFocus?: (ruleId: string) => void
  onClearPathFocus?: () => void
}

type FindingRecord = Record<string, unknown>

function stringValue(value: unknown): string | null {
  const normalized = String(value ?? '').trim()
  return normalized || null
}

function findingPath(finding: FindingRecord): string | null {
  const directPath = stringValue(finding.relative_path) ?? stringValue(finding.path) ?? stringValue(finding.filename)
  if (directPath) return directPath
  const details = finding.details && typeof finding.details === 'object' ? (finding.details as FindingRecord) : null
  const matchedDocuments = Array.isArray(details?.matched_documents)
    ? (details.matched_documents as FindingRecord[])
    : []
  return stringValue(matchedDocuments[0]?.relative_path) ?? stringValue(matchedDocuments[0]?.filename)
}

function statusLabel(status: string): string {
  return status === 'fail' ? '失败' : status === 'pass' ? '通过' : '需人工复查'
}

function RegulatoryProvenance({ finding }: { finding: FindingRecord }) {
  const details = finding.details && typeof finding.details === 'object' ? (finding.details as FindingRecord) : null
  const provenance = details?.regulatory_provenance && typeof details.regulatory_provenance === 'object'
    ? (details.regulatory_provenance as FindingRecord)
    : null
  if (!provenance) return null

  const sourceFile = stringValue(provenance.source_filename) ?? '未解析来源文件'
  const section = stringValue(provenance.section)
  const heading = stringValue(provenance.section_heading)
  const page = Number(provenance.source_page)
  const description = stringValue(provenance.rule_description)
  const excerpt = stringValue(provenance.source_excerpt)
  const anchor = stringValue(provenance.citation_anchor)
  const status = stringValue(provenance.traceability_status)
  const exact = status === 'exact_clause'

  return (
    <Collapse
      size="small"
      items={[{
        key: 'regulatory-provenance',
        label: exact ? '法规依据' : '法规依据（需人工确认来源）',
        children: (
          <Space direction="vertical" size={4} style={{ width: '100%' }}>
            <Typography.Text type="secondary">
              来源：{sourceFile}{section ? ` · 第 ${section} 节` : ''}{Number.isFinite(page) && page > 0 ? ` · PDF 第 ${page} 页` : ''}
            </Typography.Text>
            {heading ? <Typography.Text strong>{heading}</Typography.Text> : null}
            {description ? <Typography.Paragraph className="rule-detail-line" style={{ marginBottom: 0 }}>{description}</Typography.Paragraph> : null}
            {excerpt && excerpt !== description ? (
              <Typography.Paragraph type="secondary" className="rule-detail-line" style={{ marginBottom: 0 }}>
                原文摘录：{excerpt}
              </Typography.Paragraph>
            ) : null}
            {anchor ? <Typography.Text type="secondary" copyable={{ text: anchor }}>引用锚点：{anchor}</Typography.Text> : null}
          </Space>
        ),
      }]}
    />
  )
}

function statusColor(status: string): string {
  return status === 'fail' ? 'red' : status === 'pass' ? 'green' : 'orange'
}

function layerStatusColor(layer: ProjectFindingLayer): string {
  if (layer.status === 'actionable') return layer.counts.fail > 0 ? 'red' : 'orange'
  if (layer.status === 'clear') return 'green'
  return 'default'
}

function layerStatusLabel(layer: ProjectFindingLayer): string {
  if (layer.status === 'actionable') return '存在需处理项'
  if (layer.status === 'clear') return '已检查通过'
  return '暂不可用'
}

function FindingList({
  findings,
  onPathSelect,
  onRuleFocus,
}: {
  findings: FindingRecord[]
  onPathSelect?: (path: string) => void
  onRuleFocus?: (ruleId: string) => void
}) {
  if (findings.length === 0) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="当前没有需要处理的项目" />
  }

  return (
    <List
      size="small"
      dataSource={findings}
      renderItem={(finding, index) => {
        const ruleId = stringValue(finding.rule_id)
        const path = findingPath(finding)
        const message = stringValue(finding.message) ?? '未提供详细说明'
        const normalizedStatus = stringValue(finding.normalized_status) ?? 'review'
        const category = stringValue(finding.category)
        return (
          <List.Item key={`${ruleId ?? 'finding'}-${path ?? 'scope'}-${index}`}>
            <Space direction="vertical" size={4} style={{ width: '100%' }}>
              <Space wrap size={[6, 4]}>
                <Tag color={statusColor(normalizedStatus)} icon={<ExclamationCircleOutlined />}>
                  {statusLabel(normalizedStatus)}
                </Tag>
                {ruleId ? <Tag>{ruleId}</Tag> : null}
                {category ? <Tag color="blue">{category}</Tag> : null}
              </Space>
              <Typography.Text className="rule-detail-line">{message}</Typography.Text>
              <PdfPresentationContract
                contract={
                  finding.details && typeof finding.details === 'object'
                    ? ((finding.details as FindingRecord).pdf_presentation_contract as Record<string, unknown> | undefined)
                    : null
                }
              />
              <Space wrap size={[4, 4]}>
                {path && onPathSelect ? (
                  <Button
                    type="link"
                    size="small"
                    icon={<LinkOutlined />}
                    onClick={() => onPathSelect(path)}
                    style={{ paddingInline: 0, height: 'auto' }}
                  >
                    {path}
                  </Button>
                ) : path ? (
                  <Typography.Text type="secondary">位置：{path}</Typography.Text>
                ) : null}
                {ruleId && onRuleFocus ? (
                  <Button
                    type="link"
                    size="small"
                    icon={<FileSearchOutlined />}
                    onClick={() => onRuleFocus(ruleId)}
                    style={{ paddingInline: 0, height: 'auto' }}
                  >
                    查看规则证据
                  </Button>
                ) : null}
              </Space>
              <RegulatoryProvenance finding={finding} />
            </Space>
          </List.Item>
        )
      }}
    />
  )
}

function ProjectFindingLayerCard({
  layer,
  onPathSelect,
  onRuleFocus,
}: {
  layer: ProjectFindingLayer
  onPathSelect?: (path: string) => void
  onRuleFocus?: (ruleId: string) => void
}) {
  const summary = [
    layer.counts.fail > 0 ? `失败 ${layer.counts.fail}` : null,
    layer.counts.review > 0 ? `需复查 ${layer.counts.review}` : null,
    layer.counts.pass > 0 ? `通过 ${layer.counts.pass}` : null,
  ].filter(Boolean)

  return (
    <div className="project-finding-layer">
      <Space direction="vertical" size={8} style={{ width: '100%' }}>
        <Space wrap>
          <Typography.Text strong>{layer.title}</Typography.Text>
          <Tag color={layerStatusColor(layer)} icon={layer.status === 'clear' ? <CheckCircleOutlined /> : undefined}>
            {layerStatusLabel(layer)}
          </Tag>
          {summary.map((item) => <Tag key={item}>{item}</Tag>)}
        </Space>
        <Typography.Text type="secondary" className="rule-detail-line">
          {layer.description}
        </Typography.Text>
        {layer.actionable.length > 0 ? (
          <FindingList findings={layer.actionable} onPathSelect={onPathSelect} onRuleFocus={onRuleFocus} />
        ) : (
          <Typography.Text type="secondary">当前没有失败或人工复查项目。</Typography.Text>
        )}
        {layer.passed.length > 0 ? (
          <Collapse
            size="small"
            items={[
              {
                key: `${layer.id}-passed`,
                label: `查看已通过项目（${layer.passed.length}）`,
                children: <FindingList findings={layer.passed} onPathSelect={onPathSelect} onRuleFocus={onRuleFocus} />,
              },
            ]}
          />
        ) : null}
      </Space>
    </div>
  )
}

export function ProjectFindingLayersPanel({
  workbench,
  selectedPath = null,
  onPathSelect,
  onRuleFocus,
  onClearPathFocus,
}: ProjectFindingLayersPanelProps) {
  const result = buildProjectFindingLayers(workbench, { selectedPath })
  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Alert
        type={result.actionableCount > 0 ? 'warning' : 'success'}
        showIcon
        message={result.actionableCount > 0 ? `项目审核有 ${result.actionableCount} 项需要处理` : '项目审核未发现需优先处理的问题'}
        description="默认只显示失败和人工复查项目；已通过项目仅统计，展开后可查看详细记录。"
      />
      {selectedPath ? (
        <Space wrap size={[8, 8]}>
          <Tag color="blue">当前定位：{selectedPath}</Tag>
          {onClearPathFocus ? <Button size="small" onClick={onClearPathFocus}>显示全部问题</Button> : null}
        </Space>
      ) : null}
      {result.layers.map((layer) => (
        <ProjectFindingLayerCard
          key={layer.id}
          layer={layer}
          onPathSelect={onPathSelect}
          onRuleFocus={onRuleFocus}
        />
      ))}
    </Space>
  )
}

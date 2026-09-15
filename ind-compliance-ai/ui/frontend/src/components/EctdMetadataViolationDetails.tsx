import React from 'react'
import { Space, Typography, Tag, Descriptions, Collapse, Alert } from 'antd'
import { WarningOutlined, CloseCircleOutlined } from '@ant-design/icons'
import type { RuleCheckItem } from '../types'
import {
  buildEctdMetadataLifecycleDetails,
  buildViolationDetailDisplay,
  buildCrossModuleConsistencyDetails,
  buildCrossModuleInconsistencyDisplay,
} from '../ruleDetailPresentation'

const { Text, Paragraph } = Typography
const { Panel } = Collapse

interface EctdMetadataViolationDetailsProps {
  rule: RuleCheckItem
}

/**
 * 渲染 HR-ECTD-200 (元数据生命周期耦合) 的违规详情
 */
function renderMetadataLifecycleViolations(details: any) {
  const violations = details.violations || []
  const summaryRows = buildEctdMetadataLifecycleDetails(details)

  if (violations.length === 0) {
    return (
      <Alert
        type="success"
        message="未发现元数据生命周期耦合违规"
        description="所有section的元数据变更均正确更新了对应的内容文件"
        showIcon
      />
    )
  }

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      {/* 统计摘要 */}
      {summaryRows.length > 0 && (
        <Descriptions size="small" bordered column={2}>
          {summaryRows.map((row, index) => (
            <Descriptions.Item key={index} label={row.label}>
              {row.kind === 'tag' ? (
                <Tag color={row.color}>{row.value}</Tag>
              ) : (
                <Text>{row.value}</Text>
              )}
            </Descriptions.Item>
          ))}
        </Descriptions>
      )}

      {/* 违规详情列表 */}
      <Collapse defaultActiveKey={violations.length === 1 ? ['0'] : []}>
        {violations.map((violation: any, index: number) => {
          const display = buildViolationDetailDisplay(violation)

          return (
            <Panel
              key={String(index)}
              header={
                <Space>
                  <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
                  <Text strong>{display.sectionIdentifier}</Text>
                  <Tag color="red">违规</Tag>
                </Space>
              }
            >
              <Space direction="vertical" size={12} style={{ width: '100%' }}>
                {/* 违规类型 */}
                <Alert
                  type="error"
                  message={display.message}
                  showIcon
                />

                {/* 基本信息 */}
                <Descriptions size="small" bordered column={1}>
                  <Descriptions.Item label="Section标识符">
                    <Text code>{display.sectionIdentifier}</Text>
                  </Descriptions.Item>
                  <Descriptions.Item label="序列变更">
                    {display.previousSequenceNumber} → {display.sequenceNumber}
                  </Descriptions.Item>
                  <Descriptions.Item label="变更的属性">
                    <Space wrap>
                      {display.changedAttributes.map((attr: string) => (
                        <Tag key={attr} color="orange">{attr}</Tag>
                      ))}
                    </Space>
                  </Descriptions.Item>
                </Descriptions>

                {/* 元数据变更详情 */}
                {display.metadataChanges.length > 0 && (
                  <div>
                    <Text strong>元数据变更详情：</Text>
                    <Descriptions size="small" bordered column={1} style={{ marginTop: 8 }}>
                      {display.metadataChanges.map((change: any, idx: number) => (
                        <Descriptions.Item key={idx} label={change.attribute}>
                          <Space direction="vertical" size={4}>
                            <Text type="secondary">旧值: <Text delete>{change.oldValue}</Text></Text>
                            <Text type="success">新值: <Text strong>{change.newValue}</Text></Text>
                            <Tag>{change.changeType}</Tag>
                          </Space>
                        </Descriptions.Item>
                      ))}
                    </Descriptions>
                  </div>
                )}

                {/* Leaf操作详情 */}
                {display.leafOperations.length > 0 && (
                  <div>
                    <Text strong>Leaf文件操作：</Text>
                    {display.leafOperations.map((op: any, idx: number) => (
                      <Descriptions key={idx} size="small" bordered column={1} style={{ marginTop: 8 }}>
                        <Descriptions.Item label="Leaf ID">
                          <Text code>{op.leafId}</Text>
                        </Descriptions.Item>
                        <Descriptions.Item label="操作">
                          <Tag color={op.operation === 'replace' ? 'blue' : 'default'}>
                            {op.operation}
                          </Tag>
                        </Descriptions.Item>
                        <Descriptions.Item label="文件路径">
                          <Text type="secondary">{op.filePath}</Text>
                        </Descriptions.Item>
                      </Descriptions>
                    ))}
                  </div>
                )}
              </Space>
            </Panel>
          )
        })}
      </Collapse>
    </Space>
  )
}

/**
 * 渲染 HR-ECTD-201 (跨模块一致性) 的不一致详情
 */
function renderCrossModuleInconsistencies(details: any) {
  const inconsistencies = details.inconsistencies || []
  const summaryRows = buildCrossModuleConsistencyDetails(details)

  if (inconsistencies.length === 0) {
    return (
      <Alert
        type="success"
        message="未发现跨模块不一致"
        description="所有配对section的关键属性均保持一致"
        showIcon
      />
    )
  }

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      {/* 统计摘要 */}
      {summaryRows.length > 0 && (
        <Descriptions size="small" bordered column={2}>
          {summaryRows.map((row, index) => (
            <Descriptions.Item key={index} label={row.label}>
              {row.kind === 'tag' ? (
                <Tag color={row.color}>{row.value}</Tag>
              ) : (
                <Text>{row.value}</Text>
              )}
            </Descriptions.Item>
          ))}
        </Descriptions>
      )}

      {/* 不一致详情列表 */}
      <Collapse defaultActiveKey={inconsistencies.length === 1 ? ['0'] : []}>
        {inconsistencies.map((inconsistency: any, index: number) => {
          const display = buildCrossModuleInconsistencyDisplay(inconsistency)

          return (
            <Panel
              key={String(index)}
              header={
                <Space>
                  <WarningOutlined style={{ color: '#fa8c16' }} />
                  <Text strong>模块配对不一致</Text>
                  <Tag color="orange">{display.inconsistentAttributes.length} 个属性</Tag>
                </Space>
              }
            >
              <Space direction="vertical" size={12} style={{ width: '100%' }}>
                {/* 不一致说明 */}
                <Alert
                  type="warning"
                  message={display.message}
                  showIcon
                />

                {/* 模块配对信息 */}
                <Descriptions size="small" bordered column={1}>
                  <Descriptions.Item label="序列号">
                    {display.sequenceNumber}
                  </Descriptions.Item>
                  <Descriptions.Item label="模块1 Section">
                    <Text code>{display.module1Section}</Text>
                  </Descriptions.Item>
                  <Descriptions.Item label="模块2 Section">
                    <Text code>{display.module2Section}</Text>
                  </Descriptions.Item>
                </Descriptions>

                {/* 不一致属性详情 */}
                {display.inconsistentAttributes.length > 0 && (
                  <div>
                    <Text strong>不一致的属性：</Text>
                    <Descriptions size="small" bordered column={1} style={{ marginTop: 8 }}>
                      {display.inconsistentAttributes.map((attr: any, idx: number) => (
                        <Descriptions.Item key={idx} label={attr.attribute}>
                          <Space direction="vertical" size={4} style={{ width: '100%' }}>
                            <Text>
                              模块1: <Text mark>{attr.value1}</Text>
                            </Text>
                            <Text>
                              模块2: <Text mark>{attr.value2}</Text>
                            </Text>
                          </Space>
                        </Descriptions.Item>
                      ))}
                    </Descriptions>
                  </div>
                )}
              </Space>
            </Panel>
          )
        })}
      </Collapse>
    </Space>
  )
}

/**
 * 主组件：根据规则类型渲染相应的详情
 */
export function EctdMetadataViolationDetails({ rule }: EctdMetadataViolationDetailsProps) {
  const { rule_id, details } = rule

  if (!details) {
    return null
  }

  // HR-ECTD-200: 元数据生命周期耦合
  if (rule_id === 'HR-ECTD-200') {
    return (
      <div className="ectd-metadata-violation-details">
        {renderMetadataLifecycleViolations(details)}
      </div>
    )
  }

  // HR-ECTD-201: 跨模块一致性
  if (rule_id === 'HR-ECTD-201') {
    return (
      <div className="ectd-cross-module-inconsistency-details">
        {renderCrossModuleInconsistencies(details)}
      </div>
    )
  }

  return null
}


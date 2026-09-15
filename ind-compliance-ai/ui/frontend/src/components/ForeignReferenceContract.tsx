import { Alert, Collapse, Space, Tag, Typography, Divider } from 'antd'

import { buildForeignReferenceContractDisplay } from '../foreignReferenceContract'

interface ForeignReferenceContractProps {
  contract?: Record<string, unknown> | null
}

export function ForeignReferenceContract({ contract }: ForeignReferenceContractProps) {
  const display = buildForeignReferenceContractDisplay(contract)
  if (!display) return null

  return (
    <Collapse
      size="small"
      items={[{
        key: 'foreign-reference-contract',
        label: '外文参考资料规范与审核边界',
        children: (
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            {display.schemaVersion ? <Tag color="blue">契约 {display.schemaVersion}</Tag> : null}

            {/* 来源依据 */}
            <div>
              <Typography.Text strong>来源依据</Typography.Text>
              <Space direction="vertical" size={2} style={{ width: '100%', marginTop: 4 }}>
                {display.sourceRows.map((source) => (
                  <Typography.Text key={source.key} type="secondary" className="rule-detail-line">
                    {source.label}: {source.filename || '未提供'}
                    {source.section ? `，第 ${source.section} 节` : ''}
                    {source.heading ? ` - ${source.heading}` : ''}
                  </Typography.Text>
                ))}
              </Space>
            </div>

            <Divider style={{ margin: '8px 0' }} />

            {/* 语言分类规则 */}
            <div>
              <Typography.Text strong>语言分类规则</Typography.Text>
              <Space direction="vertical" size={4} style={{ width: '100%', marginTop: 4 }}>
                {display.languageRules.boundary ? (
                  <Typography.Text type="secondary" className="rule-detail-line">
                    {display.languageRules.boundary}
                  </Typography.Text>
                ) : null}

                <div style={{ marginTop: 4 }}>
                  <Typography.Text type="secondary">
                    <strong>中文档案标识</strong>: {display.languageRules.chineseIndicators}
                  </Typography.Text>
                  <br />
                  <Typography.Text type="secondary" style={{ fontSize: '12px' }}>
                    {display.languageRules.chineseDescription}
                  </Typography.Text>
                </div>

                <div style={{ marginTop: 4 }}>
                  <Typography.Text type="secondary">
                    <strong>外文参考资料标识</strong>: {display.languageRules.foreignPattern}
                  </Typography.Text>
                  <br />
                  <Typography.Text type="secondary" style={{ fontSize: '12px' }}>
                    {display.languageRules.foreignDescription}
                  </Typography.Text>
                </div>

                {display.languageRules.iso639Standard ? (
                  <Typography.Text type="secondary" style={{ fontSize: '12px' }}>
                    验证标准: {display.languageRules.iso639Standard}
                    （支持 {display.languageRules.iso639Count} 种语言代码）
                  </Typography.Text>
                ) : null}
              </Space>
            </div>

            <Divider style={{ margin: '8px 0' }} />

            {/* 结构要求 */}
            <div>
              <Typography.Text strong>结构要求</Typography.Text>
              <Space direction="vertical" size={4} style={{ width: '100%', marginTop: 4 }}>
                {display.structureRules.colocationRequirement ? (
                  <div>
                    <Typography.Text type="secondary">
                      <strong>同级配置</strong>: {display.structureRules.colocationRequirement}
                    </Typography.Text>
                    <br />
                    <Typography.Text type="secondary" style={{ fontSize: '12px' }}>
                      {display.structureRules.colocationRationale}
                    </Typography.Text>
                  </div>
                ) : null}

                {display.structureRules.orderingRequirement ? (
                  <div style={{ marginTop: 4 }}>
                    <Typography.Text type="secondary">
                      <strong>顺序约束</strong>: {display.structureRules.orderingRequirement}
                    </Typography.Text>
                    <br />
                    <Typography.Text type="secondary" style={{ fontSize: '12px' }}>
                      {display.structureRules.orderingRationale}
                    </Typography.Text>
                  </div>
                ) : null}
              </Space>
            </div>

            <Divider style={{ margin: '8px 0' }} />

            {/* 生命周期一致性 */}
            <div>
              <Typography.Text strong>生命周期一致性</Typography.Text>
              <Space direction="vertical" size={4} style={{ width: '100%', marginTop: 4 }}>
                {display.lifecycleRules.replaceRule ? (
                  <Typography.Text type="secondary">
                    <strong>Replace 操作规则</strong>: {display.lifecycleRules.replaceRule}
                  </Typography.Text>
                ) : null}
                {display.lifecycleRules.rationale ? (
                  <Typography.Text type="secondary" style={{ fontSize: '12px' }}>
                    {display.lifecycleRules.rationale}
                  </Typography.Text>
                ) : null}
                {display.lifecycleRules.failureCondition ? (
                  <Typography.Text type="secondary" style={{ fontSize: '12px' }}>
                    <strong>违规条件</strong>: {display.lifecycleRules.failureCondition}
                  </Typography.Text>
                ) : null}
              </Space>
            </div>

            <Divider style={{ margin: '8px 0' }} />

            {/* 自动化边界 */}
            {display.deterministicChecks.length > 0 ? (
              <div>
                <Space wrap size={[4, 4]}>
                  <Tag color="green">已自动核查</Tag>
                  {display.deterministicChecks.map((check, idx) => (
                    <Tag key={idx}>{check}</Tag>
                  ))}
                </Space>
              </div>
            ) : null}

            {display.hasManualReview ? (
              <Alert
                type="warning"
                showIcon
                message="以下项目需要人工复核"
                description={
                  <ul style={{ margin: 0, paddingLeft: 18 }}>
                    {display.manualReviewItems.map((item, idx) => <li key={idx}>{item}</li>)}
                  </ul>
                }
              />
            ) : null}
          </Space>
        ),
      }]}
    />
  )
}

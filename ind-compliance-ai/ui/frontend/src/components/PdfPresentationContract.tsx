import { Alert, Collapse, Space, Tag, Typography } from 'antd'

import { buildPdfPresentationContractDisplay } from '../pdfPresentationContract.js'

interface PdfPresentationContractProps {
  contract?: Record<string, unknown> | null
}

export function PdfPresentationContract({ contract }: PdfPresentationContractProps) {
  const display = buildPdfPresentationContractDisplay(contract)
  if (!display) return null

  return (
    <Collapse
      size="small"
      items={[{
        key: 'pdf-presentation-contract',
        label: 'PDF 版式规范与审核边界',
        children: (
          <Space direction="vertical" size={6} style={{ width: '100%' }}>
            {display.schemaVersion ? <Tag color="blue">契约 {display.schemaVersion}</Tag> : null}
            <Typography.Text strong>来源依据</Typography.Text>
            <Space direction="vertical" size={2} style={{ width: '100%' }}>
              {display.sourceRows.map((source) => (
                <Typography.Text key={source.key} type="secondary" className="rule-detail-line">
                  {source.label}: {source.filename || '未提供'}
                  {source.section ? `，第 ${source.section} 节` : source.sections ? `，第 ${source.sections} 节` : ''}
                  {source.pageLabel ? `，${source.pageLabel}` : ''}
                </Typography.Text>
              ))}
            </Space>
            {display.deterministicLabels.length > 0 ? (
              <Space wrap size={[4, 4]}>
                <Tag color="green">已自动核查</Tag>
                {display.deterministicLabels.map((label) => <Tag key={label}>{label}</Tag>)}
              </Space>
            ) : null}
            {display.hasManualReview ? (
              <Alert
                type="warning"
                showIcon
                message="以下项目需要人工复核"
                description={
                  <ul style={{ margin: 0, paddingLeft: 18 }}>
                    {display.manualReviewItems.map((item) => <li key={item}>{item}</li>)}
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

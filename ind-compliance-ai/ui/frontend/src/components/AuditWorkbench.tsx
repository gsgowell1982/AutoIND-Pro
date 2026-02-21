import { DownloadOutlined } from '@ant-design/icons'
import { ProCard } from '@ant-design/pro-components'
import { Alert, Button, Empty, List, Select, Space, Tag, Typography } from 'antd'
import ReactMarkdown from 'react-markdown'
import { useEffect, useMemo, useState } from 'react'
import { Document, Page } from 'react-pdf'

import { buildAssetUrl } from '../api'
import type { WorkbenchPayload } from '../types'

interface AuditWorkbenchProps {
  workbench: WorkbenchPayload | null
}

const PDF_RENDER_WIDTH = 420

export function AuditWorkbench({ workbench }: AuditWorkbenchProps) {
  const [currentPage, setCurrentPage] = useState<number>(1)

  const pdfDocument = workbench?.pdf_document
  const pageOptions = useMemo(
    () =>
      (pdfDocument?.pages ?? []).map((page) => ({
        label: `Page ${page.page_number}`,
        value: page.page_number,
      })),
    [pdfDocument?.pages],
  )

  useEffect(() => {
    if (pageOptions.length > 0) {
      setCurrentPage(pageOptions[0].value)
    }
  }, [pageOptions])

  const activePage = useMemo(
    () => pdfDocument?.pages.find((page) => page.page_number === currentPage) ?? null,
    [pdfDocument?.pages, currentPage],
  )

  const activeBoundingBoxes = useMemo(
    () =>
      (pdfDocument?.bounding_boxes ?? [])
        .filter((item) => item.page_number === currentPage)
        .slice(0, 60),
    [pdfDocument?.bounding_boxes, currentPage],
  )

  const ratio = activePage ? PDF_RENDER_WIDTH / activePage.width : 1
  const overlayHeight = activePage ? activePage.height * ratio : 0

  return (
    <ProCard
      title="2) 智能审核工作台 (Audit Workbench)"
      subTitle="左：高亮 PDF + Bounding Box，中：结构化 Markdown，右：AI 规则检查占位"
      bordered
      headerBordered
    >
      <ProCard split="vertical">
        <ProCard title="PDF 阅读与 Bounding Box" colSpan="36%" className="workbench-col">
          {!pdfDocument ? (
            <Empty description="本次上传未包含 PDF，无法展示高亮阅读器" />
          ) : (
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Typography.Text strong>{pdfDocument.filename}</Typography.Text>
              <Select
                value={currentPage}
                options={pageOptions}
                style={{ width: 180 }}
                onChange={setCurrentPage}
              />
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
                      <div className="bbox-overlay-layer" style={{ width: PDF_RENDER_WIDTH, height: overlayHeight }}>
                        {activeBoundingBoxes.map((item) => {
                          const boxWidth = (item.bbox.x1 - item.bbox.x0) * ratio
                          const boxHeight = (item.bbox.y1 - item.bbox.y0) * ratio
                          return (
                            <div
                              key={item.id}
                              className="bbox-rect"
                              style={{
                                left: item.bbox.x0 * ratio,
                                top: item.bbox.y0 * ratio,
                                width: boxWidth,
                                height: boxHeight,
                              }}
                              title={item.text}
                            />
                          )
                        })}
                      </div>
                    ) : null}
                  </div>
                </Document>
              </div>
              <List
                size="small"
                bordered
                dataSource={activeBoundingBoxes}
                renderItem={(item) => (
                  <List.Item>
                    <Typography.Text ellipsis={{ tooltip: item.text }} style={{ width: '100%' }}>
                      {item.text}
                    </Typography.Text>
                  </List.Item>
                )}
              />
            </Space>
          )}
        </ProCard>

        <ProCard title="结构化 Markdown 视图" colSpan="34%" className="workbench-col">
          <Space direction="vertical" size={10} style={{ width: '100%' }}>
            <Alert
              type="info"
              showIcon
              message="当前视图仅展示标准化摘要与第一页预览，完整解析请下载 Markdown 文件。"
            />
            {workbench?.full_markdown_download_url ? (
              <Button
                icon={<DownloadOutlined />}
                href={buildAssetUrl(workbench.full_markdown_download_url)}
                target="_blank"
              >
                下载完整解析 Markdown
              </Button>
            ) : null}
          </Space>
          {workbench?.markdown ? (
            <div className="markdown-panel">
              <ReactMarkdown>{workbench.markdown}</ReactMarkdown>
            </div>
          ) : (
            <Empty description="等待解析结果" />
          )}
        </ProCard>

        <ProCard title="AI 规则检查列表（预留）" colSpan="30%" className="workbench-col">
          <Space direction="vertical" style={{ width: '100%' }}>
            <Alert
              message={workbench?.rule_checks.message ?? 'Phase 1 暂不实现 AI 规则检查，仅保留 UI 位置'}
              type="info"
              showIcon
            />
            <Tag color="blue">Phase 1 Placeholder</Tag>
            <Empty description="规则检查列表将在下一阶段接入 Rule Engine" />
          </Space>
        </ProCard>
      </ProCard>
    </ProCard>
  )
}

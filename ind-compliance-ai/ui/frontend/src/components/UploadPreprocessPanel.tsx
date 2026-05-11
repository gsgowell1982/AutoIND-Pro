import { InboxOutlined, PlayCircleOutlined } from '@ant-design/icons'
import { ProCard } from '@ant-design/pro-components'
import { Button, Col, List, Progress, Row, Space, Tag, Typography, Upload } from 'antd'
import type { UploadFile, UploadProps } from 'antd/es/upload/interface'
import { useMemo, useState } from 'react'

import type { JobStatusResponse } from '../types'

interface UploadPreprocessPanelProps {
  onSubmit: (files: File[]) => Promise<void>
  processing: boolean
  jobStatus: JobStatusResponse | null
}

const statusColorMap: Record<string, string> = {
  queued: 'default',
  processing: 'processing',
  completed: 'success',
  completed_with_warnings: 'warning',
  failed: 'error',
}

const statusLabelMap: Record<string, string> = {
  queued: '等待处理',
  processing: '处理中',
  completed: '已完成',
  completed_with_warnings: '已完成，有提示',
  failed: '处理失败',
  idle: '未开始',
}

function getStatusLabel(status: string | null | undefined): string {
  return statusLabelMap[status ?? 'idle'] ?? status ?? '未开始'
}

export function UploadPreprocessPanel({
  onSubmit,
  processing,
  jobStatus,
}: UploadPreprocessPanelProps) {
  const [fileList, setFileList] = useState<UploadFile[]>([])

  const selectedFiles = useMemo(() => {
    const files: File[] = []
    fileList.forEach((item) => {
      if (item.originFileObj) {
        files.push(item.originFileObj as File)
      }
    })
    return files
  }, [fileList])

  const uploadProps: UploadProps = {
    multiple: true,
    fileList,
    beforeUpload: () => false,
    accept: '.pdf,.doc,.docx,.ppt,.pptx',
    onChange: ({ fileList: nextList }) => setFileList(nextList),
  }

  return (
    <ProCard
      title="资料上传"
      subTitle="支持 PDF、Word、PPT；上传后生成审阅工作台。"
      bordered
      headerBordered
    >
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Upload.Dragger {...uploadProps} height={220}>
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">拖拽或点击上传 IND 材料</p>
            <p className="ant-upload-hint">当前阶段优先呈现结构解析、资料清单和证据边界。</p>
          </Upload.Dragger>
          <Space style={{ marginTop: 16 }}>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              disabled={selectedFiles.length === 0 || processing}
              loading={processing}
              onClick={() => void onSubmit(selectedFiles)}
            >
              开始审阅
            </Button>
            <Typography.Text type="secondary">已选择 {selectedFiles.length} 个文件</Typography.Text>
          </Space>
        </Col>
        <Col xs={24} lg={12}>
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <div>
              <Typography.Text strong>处理状态：</Typography.Text>{' '}
              <Tag color={statusColorMap[jobStatus?.status ?? 'queued']}>{getStatusLabel(jobStatus?.status)}</Tag>
            </div>
            {jobStatus?.upload_scope_overview ? (
              <div>
                <Space wrap size={[8, 8]}>
                  <Tag color="cyan">上传预判: {jobStatus.upload_scope_overview.upload_mode_label}</Tag>
                  <Tag color="blue">
                    可能作用域: {jobStatus.upload_scope_overview.likely_scope_labels.join(' / ')}
                  </Tag>
                  <Tag>文件 {jobStatus.upload_scope_overview.file_count}</Tag>
                  <Tag>序列根 {jobStatus.upload_scope_overview.sequence_root_count}</Tag>
                  <Tag>申请根 {jobStatus.upload_scope_overview.application_root_count}</Tag>
                </Space>
                <Typography.Text type="secondary">
                  {jobStatus.upload_scope_overview.scope_notice}
                </Typography.Text>
              </div>
            ) : null}
            <Progress percent={jobStatus?.progress ?? 0} status={processing ? 'active' : undefined} />
            <List
              bordered
              size="small"
              dataSource={jobStatus?.files ?? []}
              renderItem={(item) => (
                <List.Item>
                  <Space direction="vertical" size={4} style={{ width: '100%' }}>
                    <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                      <Typography.Text>{item.relative_path || item.filename}</Typography.Text>
                      <Tag color={statusColorMap[item.status]}>{getStatusLabel(item.status)}</Tag>
                    </Space>
                    <Progress percent={item.progress} size="small" />
                    {item.message ? <Typography.Text type="secondary">{item.message}</Typography.Text> : null}
                  </Space>
                </List.Item>
              )}
            />
          </Space>
        </Col>
      </Row>
    </ProCard>
  )
}

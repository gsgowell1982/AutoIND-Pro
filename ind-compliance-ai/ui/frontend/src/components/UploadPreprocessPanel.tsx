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

export function UploadPreprocessPanel({
  onSubmit,
  processing,
  jobStatus,
}: UploadPreprocessPanelProps) {
  const [fileList, setFileList] = useState<UploadFile[]>([])

  const selectedFiles = useMemo(
    () => fileList.map((item) => item.originFileObj).filter((item): item is File => Boolean(item)),
    [fileList],
  )

  const uploadProps: UploadProps = {
    multiple: true,
    fileList,
    beforeUpload: () => false,
    accept: '.pdf,.doc,.docx,.ppt,.pptx',
    onChange: ({ fileList: nextList }) => setFileList(nextList),
  }

  return (
    <ProCard
      title="1) 文件上传与预处理区"
      subTitle="支持 PDF / Word / PPT 通用解析，并展示实时解析进度"
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
            <p className="ant-upload-hint">Phase 1 仅做通用解析，专项 IND 语义解析将在下一阶段扩展</p>
          </Upload.Dragger>
          <Space style={{ marginTop: 16 }}>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              disabled={selectedFiles.length === 0 || processing}
              loading={processing}
              onClick={() => void onSubmit(selectedFiles)}
            >
              开始解析
            </Button>
            <Typography.Text type="secondary">已选择 {selectedFiles.length} 个文件</Typography.Text>
          </Space>
        </Col>
        <Col xs={24} lg={12}>
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <div>
              <Typography.Text strong>任务状态：</Typography.Text>{' '}
              <Tag color={statusColorMap[jobStatus?.status ?? 'queued']}>{jobStatus?.status ?? 'idle'}</Tag>
            </div>
            <Progress percent={jobStatus?.progress ?? 0} status={processing ? 'active' : undefined} />
            <List
              bordered
              size="small"
              dataSource={jobStatus?.files ?? []}
              renderItem={(item) => (
                <List.Item>
                  <Space direction="vertical" size={4} style={{ width: '100%' }}>
                    <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                      <Typography.Text>{item.filename}</Typography.Text>
                      <Tag color={statusColorMap[item.status]}>{item.status}</Tag>
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

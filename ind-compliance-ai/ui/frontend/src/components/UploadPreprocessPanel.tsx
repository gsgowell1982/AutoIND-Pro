import { FolderOpenOutlined, InboxOutlined, PlayCircleOutlined, SearchOutlined, UnorderedListOutlined } from '@ant-design/icons'
import { ProCard } from '@ant-design/pro-components'
import { Button, Col, Drawer, Empty, Input, List, Progress, Row, Segmented, Space, Tag, Typography, Upload, message } from 'antd'
import type { UploadFile, UploadProps } from 'antd/es/upload/interface'
import { useMemo, useState } from 'react'

import type { JobStatusResponse } from '../types'
import { buildUploadSelectionSummary, filterUploadSelectionFiles } from '../uploadSelectionSummary'
import type { UploadSelectionFileRecord } from '../uploadSelectionSummary'
import { ProjectPackageTree } from './ProjectPackageTree'

interface UploadPreprocessPanelProps {
  onSubmit: (files: File[], directoryPaths?: string[]) => Promise<void>
  processing: boolean
  jobStatus: JobStatusResponse | null
}

interface FileSystemFileHandleLike {
  kind: 'file'
  name: string
  getFile: () => Promise<File>
}

interface FileSystemDirectoryHandleLike {
  kind: 'directory'
  name: string
  values: () => AsyncIterable<FileSystemEntryLike>
}

type FileSystemEntryLike = FileSystemFileHandleLike | FileSystemDirectoryHandleLike
type DirectoryPickerWindow = Window & {
  showDirectoryPicker?: () => Promise<FileSystemDirectoryHandleLike>
}

function getRelativePath(file: File): string {
  return (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name
}

function deriveDirectoryPaths(paths: string[]): string[] {
  const directories = new Set<string>()
  paths.forEach((path) => {
    const parts = path.replaceAll('\\', '/').split('/').filter(Boolean)
    parts.slice(0, -1).forEach((_, index) => directories.add(parts.slice(0, index + 1).join('/')))
  })
  return [...directories].sort()
}

function attachRelativePath(file: File, relativePath: string): File {
  const fileWithPath = file as File & { webkitRelativePath?: string }
  try {
    Object.defineProperty(fileWithPath, 'webkitRelativePath', {
      configurable: true,
      value: relativePath,
    })
  } catch {
    fileWithPath.webkitRelativePath = relativePath
  }
  return fileWithPath
}

async function collectDirectory(
  handle: FileSystemDirectoryHandleLike,
  prefix: string,
  files: File[],
  directories: Set<string>,
): Promise<void> {
  directories.add(prefix)
  for await (const entry of handle.values()) {
    const relativePath = `${prefix}/${entry.name}`
    if (entry.kind === 'directory') {
      await collectDirectory(entry, relativePath, files, directories)
    } else {
      files.push(attachRelativePath(await entry.getFile(), relativePath))
    }
  }
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

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
}

export function UploadPreprocessPanel({
  onSubmit,
  processing,
  jobStatus,
}: UploadPreprocessPanelProps) {
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [intakeMode, setIntakeMode] = useState<'single' | 'zip' | 'directory'>('single')
  const [submitting, setSubmitting] = useState(false)
  const [selectedDirectoryPaths, setSelectedDirectoryPaths] = useState<string[]>([])
  const [selectionDrawerOpen, setSelectionDrawerOpen] = useState(false)
  const [selectionQuery, setSelectionQuery] = useState('')

  const selectedFiles = useMemo(() => {
    const files: File[] = []
    fileList.forEach((item) => {
      if (item.originFileObj) {
        files.push(item.originFileObj as File)
      }
    })
    return files
  }, [fileList])

  const selectedProjectPaths = useMemo(
    () => selectedFiles.map(getRelativePath),
    [selectedFiles],
  )

  const selectedFileRecords = useMemo<UploadSelectionFileRecord[]>(
    () => selectedFiles.map((file) => ({ path: getRelativePath(file), name: file.name, size: file.size })),
    [selectedFiles],
  )

  const selectionSummary = useMemo(
    () =>
      buildUploadSelectionSummary({
        mode: intakeMode,
        files: selectedFileRecords,
        directoryPaths: selectedDirectoryPaths,
      }),
    [intakeMode, selectedDirectoryPaths, selectedFileRecords],
  )

  const filteredSelectedFileRecords = useMemo(
    () => filterUploadSelectionFiles(selectedFileRecords, selectionQuery),
    [selectedFileRecords, selectionQuery],
  )

  const selectedProjectTreePaths = useMemo(
    () => [...selectedDirectoryPaths.map((path) => `${path}/`), ...selectedProjectPaths],
    [selectedDirectoryPaths, selectedProjectPaths],
  )

  const directoryPickerSupported =
    typeof window !== 'undefined' && typeof (window as DirectoryPickerWindow).showDirectoryPicker === 'function'

  const handlePickCompleteDirectory = async () => {
    const picker = (window as DirectoryPickerWindow).showDirectoryPicker
    if (!picker) {
      message.warning('当前浏览器不支持完整目录清单读取；请改用 ZIP 上传以保留空目录。')
      return
    }
    try {
      const rootHandle = await picker()
      const files: File[] = []
      const directories = new Set<string>()
      await collectDirectory(rootHandle, rootHandle.name, files, directories)
      if (files.length === 0 && directories.size === 0) {
        message.warning('所选文件夹为空。')
        return
      }
      setFileList(
        files.map((file, index) => ({
          uid: `directory-${index}-${file.name}`,
          name: getRelativePath(file),
          status: 'done',
          originFileObj: file as UploadFile['originFileObj'],
        })),
      )
      setSelectedDirectoryPaths([...directories].sort())
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return
      message.error('读取项目文件夹失败，请改用 ZIP 上传或检查浏览器权限。')
    }
  }

  const uploadProps: UploadProps = {
    multiple: intakeMode === 'directory',
    directory: intakeMode === 'directory',
    fileList,
    beforeUpload: () => false,
    // Folder mode must not filter extensions: unknown files are evidence for package completeness checks.
    accept:
      intakeMode === 'zip'
        ? '.zip'
        : intakeMode === 'single'
          ? '.pdf,.doc,.docx,.ppt,.pptx,.xml'
          : undefined,
    onChange: ({ fileList: nextList }) => {
      setFileList(nextList)
      if (intakeMode === 'directory') {
        setSelectedDirectoryPaths(deriveDirectoryPaths(nextList.flatMap((item) => (item.originFileObj ? [getRelativePath(item.originFileObj)] : []))))
      }
    },
  }

  return (
    <>
    <ProCard
      title="资料上传"
      subTitle="支持 PDF、Word、PPT；上传后生成审阅工作台。"
      bordered
      headerBordered
    >
      <Row gutter={[16, 16]} align="stretch" className="upload-columns">
        <Col xs={24} lg={12} className="upload-column">
          <div className="upload-column-content upload-intake-column">
          <Space direction="vertical" size={16} className="upload-intake-controls">
          <Segmented
            block
            value={intakeMode}
            options={[
              { label: '上传 eCTD ZIP 项目包', value: 'zip' },
              { label: '选择项目文件夹', value: 'directory' },
              { label: '上传单个文件', value: 'single' },
            ]}
            onChange={(value) => {
              setIntakeMode(value as 'single' | 'zip' | 'directory')
              setFileList([])
              setSelectedDirectoryPaths([])
              setSelectionDrawerOpen(false)
              setSelectionQuery('')
            }}
          />
          {intakeMode === 'directory' ? (
            <Space direction="vertical" size={4} style={{ width: '100%' }}>
              <Button
                block
                icon={<FolderOpenOutlined />}
                onClick={() => void handlePickCompleteDirectory()}
                disabled={!directoryPickerSupported}
              >
                {directoryPickerSupported ? '选择完整项目文件夹（含空目录）' : '当前浏览器不支持完整目录选择'}
              </Button>
              <Typography.Text type="secondary">
                完整目录选择会同时发送空目录清单；兼容模式仅能读取包含文件的目录。
              </Typography.Text>
            </Space>
          ) : null}
          <div onClick={intakeMode === 'directory' ? () => void handlePickCompleteDirectory() : undefined}>
          <Upload.Dragger
            {...uploadProps}
            height={220}
            openFileDialogOnClick={intakeMode !== 'directory'}
            showUploadList={intakeMode === 'single'}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">
              {intakeMode === 'single'
                ? '拖拽或点击上传单个审阅文件'
                : intakeMode === 'zip'
                  ? '拖拽或点击上传 eCTD ZIP 项目包'
                  : '点击选择完整的 eCTD 项目文件夹'}
            </p>
            <p className="ant-upload-hint">
              {intakeMode === 'single'
                ? '支持 PDF、Word、PPT 和 XML，保留原有单文件自动解析能力。'
                : intakeMode === 'zip'
                  ? '后台自动解压并保留完整目录、文件和空目录信息。'
                  : '浏览器会上传目录内文件并保留相对路径；正式完整性审核建议使用 ZIP。'}
            </p>
          </Upload.Dragger>
          </div>
          {intakeMode !== 'single' ? (
            <div className="upload-selection-summary">
              <Space wrap size={[8, 8]}>
                <Tag color="blue">文件 {selectionSummary.fileCount}</Tag>
                <Tag>目录 {selectionSummary.directoryCount}</Tag>
                {selectionSummary.rootLabel ? <Tag color="cyan">根目录 {selectionSummary.rootLabel}</Tag> : null}
                <Tag>{formatBytes(selectionSummary.totalBytes)}</Tag>
              </Space>
              <Button
                type="link"
                size="small"
                icon={<UnorderedListOutlined />}
                disabled={selectionSummary.fileCount === 0}
                onClick={() => setSelectionDrawerOpen(true)}
              >
                查看文件清单
              </Button>
            </div>
          ) : null}
          </Space>
          <Space className="upload-submit-row">
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              disabled={selectedFiles.length === 0 || processing || submitting}
              loading={processing || submitting}
              onClick={() => {
                setSubmitting(true)
                void onSubmit(selectedFiles, intakeMode === 'directory' ? selectedDirectoryPaths : []).finally(() =>
                  setSubmitting(false),
                )
              }}
            >
              {submitting ? '正在上传...' : '开始审阅'}
            </Button>
            <Typography.Text type="secondary">已选择 {selectedFiles.length} 个文件</Typography.Text>
          </Space>
          </div>
        </Col>
        <Col xs={24} lg={12} className="upload-column">
          <div className="upload-column-content upload-status-column">
          <div className="upload-status-header">
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
          </div>
          <div className="upload-status-results">
            {jobStatus?.package_inventory && ['queued', 'processing'].includes(jobStatus.status) ? (
              <ProjectPackageTree inventory={jobStatus.package_inventory} title="完整项目结构" />
            ) : !jobStatus && intakeMode === 'directory' && selectedProjectTreePaths.length > 0 ? (
              <ProjectPackageTree fallbackPaths={selectedProjectTreePaths} title="待上传项目结构" />
            ) : null}
            <div className="upload-file-list">
              <List
                bordered
                size="small"
                dataSource={jobStatus?.files ?? []}
                pagination={
                  (jobStatus?.files.length ?? 0) > 50
                    ? { pageSize: 50, size: 'small', showSizeChanger: false, hideOnSinglePage: true }
                    : false
                }
                renderItem={(item) => (
                  <List.Item>
                    <Space direction="vertical" size={4} style={{ width: '100%', minWidth: 0 }}>
                      <Space style={{ width: '100%', justifyContent: 'space-between', minWidth: 0 }}>
                        <Typography.Text
                          ellipsis={{ tooltip: item.relative_path || item.filename }}
                          style={{ minWidth: 0, flex: 1 }}
                        >
                          {item.relative_path || item.filename}
                        </Typography.Text>
                        <Tag color={statusColorMap[item.status]}>{getStatusLabel(item.status)}</Tag>
                      </Space>
                      <Progress percent={item.progress} size="small" />
                      {item.message ? <Typography.Text type="secondary">{item.message}</Typography.Text> : null}
                    </Space>
                  </List.Item>
                )}
              />
            </div>
          </div>
          </div>
        </Col>
      </Row>
    </ProCard>
    <Drawer
      className="upload-selection-drawer"
      title={`已选择文件（${selectionSummary.fileCount}）`}
      open={selectionDrawerOpen}
      width={560}
      onClose={() => {
        setSelectionDrawerOpen(false)
        setSelectionQuery('')
      }}
    >
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        <Input
          allowClear
          prefix={<SearchOutlined />}
          placeholder="搜索文件相对路径"
          value={selectionQuery}
          onChange={(event) => setSelectionQuery(event.target.value)}
        />
        <Space wrap size={[8, 8]}>
          <Tag color="blue">文件 {selectionSummary.fileCount}</Tag>
          <Tag>目录 {selectionSummary.directoryCount}</Tag>
          <Tag>大小 {formatBytes(selectionSummary.totalBytes)}</Tag>
        </Space>
        {filteredSelectedFileRecords.length > 0 ? (
          <div className="upload-selection-file-list">
            <List
              size="small"
              dataSource={filteredSelectedFileRecords}
              renderItem={(item) => (
                <List.Item>
                  <Space style={{ width: '100%', minWidth: 0, justifyContent: 'space-between' }}>
                    <Typography.Text
                      className="upload-selection-file-path"
                      ellipsis={{ tooltip: item.path || item.name }}
                    >
                      {item.path || item.name}
                    </Typography.Text>
                    <Typography.Text type="secondary">{formatBytes(item.size ?? 0)}</Typography.Text>
                  </Space>
                </List.Item>
              )}
            />
          </div>
        ) : (
          <Empty description={selectionQuery ? '没有匹配的文件' : '尚未选择文件'} />
        )}
      </Space>
    </Drawer>
  </>
  )
}

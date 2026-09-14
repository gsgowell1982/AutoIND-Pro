import { PageContainer, ProCard, ProConfigProvider } from '@ant-design/pro-components'
import { PlayCircleOutlined } from '@ant-design/icons'
import { Alert, Button, Space, Spin, Tag, Typography, message } from 'antd'
import axios from 'axios'
import { useEffect, useRef, useState } from 'react'

import {
  RUNTIME_WARNING,
  WORKSPACE_LABEL,
  fetchConsistency,
  fetchJobStatus,
  fetchWorkbench,
  startControlledDemoSample,
  uploadFiles,
} from './api'
import { AuditWorkbench } from './components/AuditWorkbench'
import { ConsistencyBoard } from './components/ConsistencyBoard'
import { UploadPreprocessPanel } from './components/UploadPreprocessPanel'
import type { ConsistencyRow, JobStatusResponse, WorkbenchPayload } from './types'

function getJobStatusTagColor(status: string | null): string {
  if (!status) {
    return 'default'
  }
  if (status === 'completed') {
    return 'success'
  }
  if (status === 'failed') {
    return 'error'
  }
  if (status === 'queued' || status === 'processing' || status === 'completed_with_warnings') {
    return 'processing'
  }
  return 'default'
}

function getJobStatusLabel(status: string | null): string {
  switch (status) {
    case 'queued':
      return '等待处理'
    case 'processing':
      return '处理中'
    case 'completed':
      return '已完成'
    case 'completed_with_warnings':
      return '已完成，有提示'
    case 'failed':
      return '处理失败'
    default:
      return '未开始'
  }
}

function App() {
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null)
  const [workbench, setWorkbench] = useState<WorkbenchPayload | null>(null)
  const [consistencyRows, setConsistencyRows] = useState<ConsistencyRow[]>([])
  const [polling, setPolling] = useState(false)
  const [loadingWorkbench, setLoadingWorkbench] = useState(false)
  const unmountedRef = useRef(false)

  useEffect(() => {
    // React StrictMode in dev runs effect cleanup/setup twice.
    // Reset to false in setup to avoid stale "unmounted" flag.
    unmountedRef.current = false
    return () => {
      unmountedRef.current = true
    }
  }, [])

  const handleUpload = async (files: File[], directoryPaths: string[] = []) => {
    try {
      setWorkbench(null)
      setConsistencyRows([])
      setJobStatus(null)
      const result = await uploadFiles(files, directoryPaths)
      setJobId(result.job_id)
      setJobStatus(result)
      setPolling(true)
      message.success('上传成功，开始预处理与通用解析')
    } catch (error) {
      console.error(error)
      if (axios.isAxiosError(error)) {
        const backendDetail =
          (error.response?.data as { detail?: string } | undefined)?.detail ?? error.message
        message.error(`上传失败：${backendDetail}`)
      } else {
        message.error('上传失败，请检查文件格式或后端服务状态')
      }
    }
  }

  const handleStartControlledDemoSample = async () => {
    try {
      setWorkbench(null)
      setConsistencyRows([])
      setJobStatus(null)
      setLoadingWorkbench(true)
      const result = await startControlledDemoSample()
      if (unmountedRef.current) {
        return
      }
      setJobId(result.job_id)
      setJobStatus(result)
      setPolling(false)
      const [workbenchPayload, consistencyPayload] = await Promise.all([
        fetchWorkbench(result.job_id),
        fetchConsistency(result.job_id),
      ])
      if (unmountedRef.current) {
        return
      }
      setWorkbench({ ...workbenchPayload, demo_sample: result.demo_sample ?? null })
      setConsistencyRows(consistencyPayload.rows)
      message.success('演示样本已加载')
    } catch (error) {
      console.error(error)
      if (axios.isAxiosError(error)) {
        const backendDetail =
          (error.response?.data as { detail?: string } | undefined)?.detail ?? error.message
        message.error(`加载演示样本失败：${backendDetail}`)
      } else {
        message.error('加载演示样本失败，请检查后端服务状态')
      }
    } finally {
      if (!unmountedRef.current) {
        setLoadingWorkbench(false)
      }
    }
  }

  useEffect(() => {
    if (!jobId || !polling) {
      return
    }
    let requesting = false

    const loadWorkbench = async (targetJobId: string, demoSample: JobStatusResponse['demo_sample'] = null) => {
      setLoadingWorkbench(true)
      try {
        const [workbenchPayload, consistencyPayload] = await Promise.all([
          fetchWorkbench(targetJobId),
          fetchConsistency(targetJobId),
        ])
        if (unmountedRef.current) {
          return
        }
        setWorkbench({ ...workbenchPayload, demo_sample: demoSample ?? null })
        setConsistencyRows(consistencyPayload.rows)
      } catch (error) {
        console.error(error)
        if (axios.isAxiosError(error)) {
          const backendDetail =
            (error.response?.data as { detail?: string } | undefined)?.detail ?? error.message
          message.error(`加载工作台失败：${backendDetail}`)
        } else {
          message.error('解析完成，但加载工作台数据失败')
        }
      } finally {
        if (!unmountedRef.current) {
          setLoadingWorkbench(false)
        }
      }
    }

    const poll = async () => {
      if (requesting) {
        return
      }
      requesting = true
      try {
        const status = await fetchJobStatus(jobId)
        if (unmountedRef.current) {
          return
        }
        setJobStatus(status)

        if (!['queued', 'processing'].includes(status.status)) {
          setPolling(false)
          if (timer !== undefined) {
            window.clearInterval(timer)
          }
          await loadWorkbench(jobId, status.demo_sample ?? null)
        }
      } catch (error) {
        console.error(error)
        setPolling(false)
        if (axios.isAxiosError(error)) {
          const backendDetail =
            (error.response?.data as { detail?: string } | undefined)?.detail ?? error.message
          message.error(`轮询失败：${backendDetail}`)
        } else {
          message.error('轮询任务状态失败，请稍后重试')
        }
      } finally {
        requesting = false
      }
    }

    const timer = window.setInterval(() => {
      void poll()
    }, 1500)

    void poll()

    return () => {
      if (timer !== undefined) {
        window.clearInterval(timer)
      }
    }
  }, [jobId, polling])

  const effectiveWorkspace = WORKSPACE_LABEL || '当前工作区'
  const runtimeWarning = RUNTIME_WARNING.trim()
  const currentStatus = jobStatus?.status ?? (polling ? 'processing' : null)

  return (
    <ProConfigProvider hashed={false}>
      <PageContainer
        title="中国 IND 智能审阅工作台"
        subTitle="eCTD 结构校验 · 资料清单 · 证据边界"
        content="面向中国 IND 申请资料，优先呈现结构解析、前置条件、人工复核提示与内容一致性线索。"
      >
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Alert
            type="info"
            showIcon
            message="当前任务"
            description={
              <Space size={[8, 8]} wrap>
                <Tag color="default">{effectiveWorkspace}</Tag>
                <Tag color={getJobStatusTagColor(currentStatus)}>
                  {getJobStatusLabel(currentStatus)}
                </Tag>
                {jobId ? <Typography.Text type="secondary">任务编号：{jobId}</Typography.Text> : null}
                <Typography.Text type="secondary">
                  重新上传文件会生成新任务，当前结果不会自动切换到旧任务。
                </Typography.Text>
              </Space>
            }
          />
          {runtimeWarning ? (
            <Alert
              type="warning"
              showIcon
              message="当前运行环境存在能力降级"
              description={runtimeWarning}
            />
          ) : null}
          <Alert
            type="warning"
            showIcon
            message="本系统提供合规审阅支持与证据提示，不替代注册申报决策。"
          />
          <Alert
            type="info"
            showIcon
            message="受控演示样本"
            description={
              <Space direction="vertical" size={8}>
                <Typography.Text>
                  无需上传文件，直接查看法规来源、前置条件、内容一致性、演示报告和演示脚本。该样本为合成演示数据，不代表真实申报资料。
                </Typography.Text>
                <Button
                  icon={<PlayCircleOutlined />}
                  onClick={() => void handleStartControlledDemoSample()}
                  loading={loadingWorkbench}
                  disabled={polling}
                >
                  加载演示样本
                </Button>
              </Space>
            }
          />
          <UploadPreprocessPanel
            onSubmit={handleUpload}
            processing={polling}
            jobStatus={jobStatus}
          />
          {loadingWorkbench ? (
            <ProCard bordered>
              <Space>
                <Spin />
                <Typography.Text>正在加载审核工作台...</Typography.Text>
              </Space>
            </ProCard>
          ) : (
            <>
              <AuditWorkbench
                key={`${jobId ?? 'empty'}:${workbench ? 'ready' : 'empty'}`}
                workbench={workbench}
              />
              <ConsistencyBoard rows={consistencyRows} />
            </>
          )}
        </Space>
      </PageContainer>
    </ProConfigProvider>
  )
}

export default App

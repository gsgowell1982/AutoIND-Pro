import { PageContainer, ProCard, ProConfigProvider } from '@ant-design/pro-components'
import { Alert, Space, Spin, Typography, message } from 'antd'
import axios from 'axios'
import { useEffect, useRef, useState } from 'react'

import { fetchConsistency, fetchJobStatus, fetchWorkbench, uploadFiles } from './api'
import { AuditWorkbench } from './components/AuditWorkbench'
import { ConsistencyBoard } from './components/ConsistencyBoard'
import { UploadPreprocessPanel } from './components/UploadPreprocessPanel'
import type { ConsistencyRow, JobStatusResponse, WorkbenchPayload } from './types'

function App() {
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null)
  const [workbench, setWorkbench] = useState<WorkbenchPayload | null>(null)
  const [consistencyRows, setConsistencyRows] = useState<ConsistencyRow[]>([])
  const [polling, setPolling] = useState(false)
  const [loadingWorkbench, setLoadingWorkbench] = useState(false)
  const unmountedRef = useRef(false)

  useEffect(() => {
    return () => {
      unmountedRef.current = true
    }
  }, [])

  const handleUpload = async (files: File[]) => {
    try {
      setWorkbench(null)
      setConsistencyRows([])
      setJobStatus(null)
      const result = await uploadFiles(files)
      setJobId(result.job_id)
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

  useEffect(() => {
    if (!jobId || !polling) {
      return
    }
    let timer: number | undefined
    let requesting = false

    const loadWorkbench = async (targetJobId: string) => {
      setLoadingWorkbench(true)
      try {
        const [workbenchPayload, consistencyPayload] = await Promise.all([
          fetchWorkbench(targetJobId),
          fetchConsistency(targetJobId),
        ])
        if (unmountedRef.current) {
          return
        }
        setWorkbench(workbenchPayload)
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
          await loadWorkbench(jobId)
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

    void poll()
    timer = window.setInterval(() => {
      void poll()
    }, 1500)

    return () => {
      if (timer !== undefined) {
        window.clearInterval(timer)
      }
    }
  }, [jobId, polling])

  return (
    <ProConfigProvider hashed={false}>
      <PageContainer
        title="IND Compliance AI · Phase 1"
        subTitle="FastAPI + React + Ant Design Pro"
        content="目标：上传材料、展示解析进度、构建审核工作台与跨模块一致性看板（AI 规则检查暂留位置）"
      >
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Alert
            type="warning"
            showIcon
            message="声明：本系统当前仅提供合规支持与可追溯信息，不替代注册申报决策。"
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
              <AuditWorkbench workbench={workbench} />
              <ConsistencyBoard rows={consistencyRows} />
            </>
          )}
        </Space>
      </PageContainer>
    </ProConfigProvider>
  )
}

export default App

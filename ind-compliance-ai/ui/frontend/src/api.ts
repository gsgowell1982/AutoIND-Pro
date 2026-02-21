import axios from 'axios'

import type {
  ConsistencyResponse,
  JobStatusResponse,
  UploadJobResponse,
  WorkbenchPayload,
} from './types'

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
})

export async function uploadFiles(files: File[]): Promise<UploadJobResponse> {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))
  // Let browser set multipart boundary automatically.
  const { data } = await client.post<UploadJobResponse>('/api/v1/uploads', formData)
  return data
}

export async function fetchJobStatus(jobId: string): Promise<JobStatusResponse> {
  const { data } = await client.get<JobStatusResponse>(`/api/v1/jobs/${jobId}`)
  return data
}

export async function fetchWorkbench(jobId: string): Promise<WorkbenchPayload> {
  const { data } = await client.get<WorkbenchPayload>(`/api/v1/jobs/${jobId}/workbench`)
  return data
}

export async function fetchConsistency(jobId: string): Promise<ConsistencyResponse> {
  const { data } = await client.get<ConsistencyResponse>(`/api/v1/jobs/${jobId}/consistency`)
  return data
}

export function buildAssetUrl(path: string): string {
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path
  }
  if (!API_BASE_URL) {
    return path
  }
  return `${API_BASE_URL}${path}`
}

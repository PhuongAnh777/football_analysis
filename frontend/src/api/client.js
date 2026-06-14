import axios from 'axios'

/** Remote Colab backend: set VITE_API_BASE=https://xxxx.ngrok-free.app/api in .env.local */
export const API_BASE = import.meta.env.VITE_API_BASE || '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'ngrok-skip-browser-warning': 'true',
  },
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const message =
      err.response?.data?.detail ||
      err.response?.data?.message ||
      err.message ||
      'Đã xảy ra lỗi không xác định'
    return Promise.reject(new Error(message))
  },
)

export function getVideoUrl(jobId) {
  return `${API_BASE}/video/${jobId}?v=${encodeURIComponent(jobId)}`
}

export async function uploadVideo(file, onProgress, { team1Name, team2Name } = {}) {
  const form = new FormData()
  form.append('video', file)
  if (team1Name?.trim()) form.append('team1_name', team1Name.trim())
  if (team2Name?.trim()) form.append('team2_name', team2Name.trim())
  const { data } = await api.post('/analyze', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 600_000,
    onUploadProgress(e) {
      if (e.total) onProgress?.(Math.round((e.loaded * 100) / e.total))
    },
  })
  return data
}

export async function getStatus(jobId) {
  const { data } = await api.get(`/status/${jobId}`)
  return data
}

export async function getResults(jobId) {
  const { data } = await api.get(`/results/${jobId}`)
  return data
}

export default api

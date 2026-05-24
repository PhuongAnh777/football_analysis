import axios from 'axios'

const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
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

export async function uploadVideo(file, onProgress) {
  const form = new FormData()
  form.append('video', file)
  const { data } = await api.post('/analyze', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
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

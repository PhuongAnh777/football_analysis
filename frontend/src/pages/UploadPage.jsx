import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useAnalysis } from '../context/AnalysisContext'
import { uploadVideo, getStatus, getResults } from '../api/client'
import { usePolling } from '../hooks/usePolling'

const PROCESSING_STEPS = [
  { key: 'reading',   label: 'Đọc video' },
  { key: 'tracking',  label: 'Tracking cầu thủ' },
  { key: 'camera',    label: 'Phân tích camera' },
  { key: 'teams',     label: 'Gán đội hình' },
  { key: 'speed',     label: 'Tốc độ & khoảng cách' },
  { key: 'tactical',  label: 'Phân tích chiến thuật' },
  { key: 'report',    label: 'Tạo báo cáo' },
  { key: 'render',    label: 'Render output' },
]

function formatBytes(bytes) {
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1048576).toFixed(1)} MB`
}

function StepIndicator({ steps, currentStepKey }) {
  const currentIdx = steps.findIndex(s => s.key === currentStepKey)
  return (
    <div className="flex flex-col gap-1 w-full max-w-xs mx-auto">
      {steps.map((step, i) => {
        const isDone = currentIdx > i, isCurrent = currentIdx === i
        return (
          <div key={step.key}>
            <div className="flex items-center gap-3 py-1.5">
              <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold border transition-all ${
                isDone    ? 'bg-emerald-500/20 border-emerald-500 text-emerald-400' :
                isCurrent ? 'bg-blue-500/20 border-blue-500 text-blue-400' :
                            'bg-surface-2 border-border text-text-secondary'}`}>
                {isDone ? '✓' : isCurrent ? (
                  <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                ) : <span>{i + 1}</span>}
              </div>
              <span className={`text-sm font-medium transition-colors ${isDone ? 'text-emerald-400' : isCurrent ? 'text-text-primary' : 'text-text-secondary'}`}>
                {step.label}
              </span>
              {isCurrent && (
                <motion.span animate={{ opacity: [1, 0.4, 1] }} transition={{ duration: 1.2, repeat: Infinity }}
                  className="ml-auto text-xs text-blue-400 font-medium">đang xử lý...</motion.span>
              )}
            </div>
            {i < steps.length - 1 && (
              <div className={`w-0.5 h-3 ml-3.5 rounded ${isDone ? 'bg-emerald-500/40' : 'bg-border'}`} />
            )}
          </div>
        )
      })}
    </div>
  )
}

export default function UploadPage() {
  const navigate = useNavigate()
  const { jobId, status, uploadProgress, processingProgress, currentStep, message,
          error, errorLogPath, setUploading, setJob, setProcessing, setDone, setError, reset } = useAnalysis()
  const [file, setFile] = useState(null)
  const [toast, setToast] = useState(null)
  const [copied, setCopied] = useState(false)
  const [team1Name, setTeam1Name] = useState('')
  const [team2Name, setTeam2Name] = useState('')

  const isUploading  = status === 'uploading'
  const isProcessing = status === 'processing'

  function showToast(msg, type = 'error') {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 4000)
  }

  usePolling(async () => {
    if (!jobId) return
    try {
      const data = await getStatus(jobId)
      setProcessing({
        progress: Math.round((data.progress ?? 0) * 100),
        currentStep: data.step_key || data.current_step,
        message: data.current_step,
      })
      if (data.status === 'done' || data.status === 'completed') {
        const results = await getResults(jobId)
        if (results.input_md5) {
          console.info('[analyze] input', results.input_filename, results.input_md5)
        }
        setDone(results)
        navigate('/dashboard')
      } else if (data.status === 'error' || data.status === 'failed') {
        const errText = data.error || data.message || 'Xử lý thất bại'
        setError(errText, data.error_log_path || null)
        showToast('Xử lý thất bại — xem log bên dưới')
      }
    } catch { /* ignore */ }
  }, 2000, isProcessing)

  const onDrop = useCallback((accepted, rejected) => {
    if (rejected.length) { showToast('File không hợp lệ. Chỉ hỗ trợ MP4 và AVI, tối đa 500MB.'); return }
    if (accepted.length) setFile(accepted[0])
  }, [])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: { 'video/mp4': ['.mp4'], 'video/x-msvideo': ['.avi'] },
    maxSize: 500 * 1024 * 1024,
    multiple: false,
    disabled: isUploading || isProcessing,
  })

  async function copyError() {
    if (!error) return
    try {
      await navigator.clipboard.writeText(error)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      showToast('Không copy được — mở file log trên máy')
    }
  }

  async function handleAnalyze() {
    if (!file) return
    try {
      reset()
      setUploading(0)
      const { job_id } = await uploadVideo(file, pct => setUploading(pct), { team1Name, team2Name })
      setJob(job_id)
    } catch (err) {
      setError(err.message)
      showToast(err.message)
    }
  }

  if (isProcessing) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8">
        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
          className="card max-w-md w-full text-center space-y-6 glow-team1">
          <div className="flex justify-center">
            <div className="w-20 h-20 rounded-full flex items-center justify-center"
              style={{ background: 'rgba(59,130,246,0.1)', border: '2px solid rgba(59,130,246,0.3)' }}>
              <span className="text-4xl football-spinner">⚽</span>
            </div>
          </div>
          <div>
            <h2 className="text-xl font-bold text-text-primary">Đang phân tích video</h2>
            <p className="text-sm text-text-secondary mt-1">{message || 'Vui lòng đợi trong giây lát...'}</p>
          </div>
          <div>
            <div className="metric-bar-track">
              <motion.div className="metric-bar-fill" style={{ background: 'var(--color-team-1)' }}
                animate={{ width: `${processingProgress}%` }} transition={{ duration: 0.6 }} />
            </div>
            <div className="flex justify-between mt-1.5 text-xs font-mono text-text-secondary">
              <span>{processingProgress}%</span>
              <span>~{Math.max(0, Math.round((100 - processingProgress) * 0.12))} giây còn lại</span>
            </div>
          </div>
          <StepIndicator steps={PROCESSING_STEPS} currentStepKey={currentStep} />
          <button onClick={() => { reset(); setFile(null) }}
            className="text-xs text-text-secondary hover:text-red-400 transition-colors underline">
            Hủy và quay lại
          </button>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="p-8 min-h-screen">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}
        className="max-w-2xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Tải lên video</h1>
          <p className="text-text-secondary mt-1">Tải lên video trận đấu để bắt đầu phân tích chiến thuật AI</p>
          {status === 'done' && (
            <p className="text-emerald-400/90 text-sm mt-2">
              Đã có kết quả phân tích trước đó — chọn video mới bên dưới để phân tích lại.
            </p>
          )}
        </div>

        {/* Drop zone */}
        <div {...getRootProps()}
          className={`upload-zone rounded-2xl p-12 text-center cursor-pointer transition-all duration-300
            ${isDragActive && !isDragReject ? 'active' : ''} ${isDragReject ? 'reject' : ''}
            ${isUploading ? 'opacity-60 cursor-not-allowed' : ''}`}>
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-4">
            <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-all ${isDragActive ? 'scale-110' : ''}`}
              style={{ background: 'var(--color-surface-2)' }}>
              <svg className={`w-8 h-8 ${isDragReject ? 'text-red-400' : 'text-blue-400'}`}
                fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
            </div>
            {isDragReject ? <p className="text-red-400 font-medium">File không được hỗ trợ</p>
            : isDragActive ? <p className="text-blue-400 font-semibold text-lg">Thả file vào đây!</p>
            : <div className="space-y-1">
                <p className="text-text-primary font-semibold text-lg">Kéo thả video vào đây hoặc click để chọn file</p>
                <p className="text-text-secondary text-sm">Hỗ trợ: <span className="text-text-primary font-medium">MP4, AVI</span> | Tối đa <span className="text-text-primary font-medium">500MB</span></p>
              </div>}
          </div>
        </div>

        {/* File preview */}
        <AnimatePresence>
          {file && (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
              className="card flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
                style={{ background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.3)' }}>
                <svg className="w-6 h-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-text-primary font-semibold text-sm truncate">{file.name}</p>
                <p className="text-text-secondary text-xs mt-0.5">{formatBytes(file.size)}</p>
              </div>
              <button onClick={(e) => { e.stopPropagation(); setFile(null) }}
                className="w-8 h-8 rounded-lg flex items-center justify-center text-text-secondary hover:text-red-400 hover:bg-red-500/10 transition-all">✕</button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Team names */}
        <div className="card space-y-3">
          <p className="text-sm font-semibold text-text-primary">Tên đội <span className="font-normal text-text-secondary">(tuỳ chọn)</span></p>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs text-blue-400 font-medium">Đội 1 (trái)</label>
              <input
                type="text"
                value={team1Name}
                onChange={e => setTeam1Name(e.target.value)}
                placeholder="Tên đội 1..."
                maxLength={40}
                disabled={isUploading || isProcessing}
                className="w-full px-3 py-2 rounded-lg text-sm bg-surface-2 border border-border text-text-primary placeholder-text-secondary focus:outline-none focus:border-blue-500/60 transition-colors disabled:opacity-40"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-red-400 font-medium">Đội 2 (phải)</label>
              <input
                type="text"
                value={team2Name}
                onChange={e => setTeam2Name(e.target.value)}
                placeholder="Tên đội 2..."
                maxLength={40}
                disabled={isUploading || isProcessing}
                className="w-full px-3 py-2 rounded-lg text-sm bg-surface-2 border border-border text-text-primary placeholder-text-secondary focus:outline-none focus:border-red-500/60 transition-colors disabled:opacity-40"
              />
            </div>
          </div>
          <p className="text-xs text-text-secondary">Nhập tên đội theo bảng tỉ số trong video để hiển thị đúng trong báo cáo.</p>
        </div>

        {/* Upload progress */}
        <AnimatePresence>
          {isUploading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="card space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-text-secondary font-medium">Đang tải lên...</span>
                <span className="font-mono font-semibold text-blue-400">{uploadProgress}%</span>
              </div>
              <div className="metric-bar-track">
                <motion.div className="metric-bar-fill" style={{ background: 'var(--color-team-1)' }}
                  animate={{ width: `${uploadProgress}%` }} transition={{ duration: 0.3 }} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Button */}
        <motion.button
          whileHover={file && !isUploading ? { scale: 1.02 } : {}}
          whileTap={file && !isUploading ? { scale: 0.98 } : {}}
          onClick={handleAnalyze}
          disabled={!file || isUploading}
          className={`w-full py-4 rounded-xl font-semibold text-base transition-all duration-200 ${
            file && !isUploading ? 'text-white cursor-pointer' : 'opacity-40 cursor-not-allowed text-text-secondary'}`}
          style={{ background: file && !isUploading
            ? 'linear-gradient(135deg, var(--color-team-1), var(--color-accent))'
            : 'var(--color-surface-2)' }}>
          {isUploading
            ? <span className="flex items-center justify-center gap-2">
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                </svg>
                Đang tải lên... {uploadProgress}%
              </span>
            : <span className="flex items-center justify-center gap-2"><span>⚡</span> Bắt đầu phân tích</span>}
        </motion.button>

        {status === 'error' && (
          <div className="card border-red-500/40 bg-red-500/5 flex flex-col gap-3">
            <div className="flex items-start gap-3">
              <span className="text-red-400 text-lg mt-0.5">⚠</span>
              <div className="flex-1 min-w-0">
                <p className="text-red-400 font-medium text-sm">Đã xảy ra lỗi</p>
                {errorLogPath && (
                  <p className="text-text-secondary text-xs mt-1 break-all">
                    Log: <code className="text-red-300/90">{errorLogPath}</code>
                  </p>
                )}
              </div>
              <div className="flex gap-2 flex-shrink-0">
                <button onClick={copyError}
                  className="text-xs px-2 py-1 rounded border border-border hover:border-red-400/50 text-text-secondary hover:text-text-primary transition-colors">
                  {copied ? 'Đã copy' : 'Copy log'}
                </button>
                <button onClick={() => { reset(); setFile(null); setCopied(false) }}
                  className="text-xs text-text-secondary hover:text-text-primary transition-colors">Thử lại</button>
              </div>
            </div>
            {error && (
              <pre className="text-xs text-red-300/90 bg-black/30 rounded-lg p-3 overflow-auto max-h-64 whitespace-pre-wrap break-words border border-red-500/20">
                {error}
              </pre>
            )}
          </div>
        )}
      </motion.div>

      <AnimatePresence>
        {toast && (
          <motion.div initial={{ x: 80, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 80, opacity: 0 }}
            className={`toast ${toast.type === 'error' ? 'bg-surface border-red-500/40 text-red-400' : 'bg-surface border-emerald-500/40 text-emerald-400'}`}>
            <span>{toast.type === 'error' ? '⚠' : '✓'}</span>
            <span>{toast.msg}</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

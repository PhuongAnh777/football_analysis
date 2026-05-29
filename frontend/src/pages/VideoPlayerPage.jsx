import { useRef, useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { useAnalysis } from '../context/AnalysisContext'
import { getVideoUrl } from '../api/client'
import Card from '../components/UI/Card'
import Badge from '../components/UI/Badge'

function formatTime(s) {
  if (!isFinite(s)) return '0:00'
  return `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, '0')}`
}

function VideoControls({ videoRef, playing, setPlaying, duration, currentTime, volume, setVolume }) {
  const pct = duration ? (currentTime / duration) * 100 : 0

  function togglePlay() {
    const v = videoRef.current; if (!v) return
    if (v.paused) { v.play(); setPlaying(true) } else { v.pause(); setPlaying(false) }
  }
  function handleSeek(e) {
    if (!videoRef.current || !duration) return
    const rect = e.currentTarget.getBoundingClientRect()
    videoRef.current.currentTime = (Math.max(0, Math.min(e.clientX - rect.left, rect.width)) / rect.width) * duration
  }
  function handleVolume(e) {
    const v = parseFloat(e.target.value); setVolume(v)
    if (videoRef.current) videoRef.current.volume = v
  }
  function toggleFS() {
    const c = videoRef.current?.parentElement?.parentElement
    if (!c) return
    document.fullscreenElement ? document.exitFullscreen() : c.requestFullscreen?.()
  }

  return (
    <div className="absolute bottom-0 left-0 right-0 px-4 py-3 rounded-b-xl"
      style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.85) 0%, transparent 100%)' }}>
      <div className="w-full h-1.5 rounded-full mb-3 cursor-pointer relative group"
        style={{ background: 'rgba(255,255,255,0.2)' }} onClick={handleSeek}>
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: 'var(--color-team-1)' }} />
      </div>
      <div className="flex items-center gap-3">
        <button onClick={togglePlay}
          className="w-8 h-8 flex items-center justify-center rounded-full text-white hover:bg-white/20 transition-colors">
          {playing
            ? <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path fillRule="evenodd" d="M6.75 5.25a.75.75 0 01.75-.75H9a.75.75 0 01.75.75v13.5a.75.75 0 01-.75.75H7.5a.75.75 0 01-.75-.75V5.25zm7.5 0A.75.75 0 0115 4.5h1.5a.75.75 0 01.75.75v13.5a.75.75 0 01-.75.75H15a.75.75 0 01-.75-.75V5.25z" clipRule="evenodd" /></svg>
            : <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path fillRule="evenodd" d="M4.5 5.653c0-1.426 1.529-2.33 2.779-1.643l11.54 6.348c1.295.712 1.295 2.573 0 3.285L7.28 19.991c-1.25.687-2.779-.217-2.779-1.643V5.653z" clipRule="evenodd" /></svg>}
        </button>
        <span className="text-xs font-mono text-white/80 tabular-nums">{formatTime(currentTime)} / {formatTime(duration)}</span>
        <div className="flex-1" />
        <button onClick={() => { const v = volume > 0 ? 0 : 0.8; setVolume(v); if (videoRef.current) videoRef.current.volume = v }}
          className="text-white/80 hover:text-white transition-colors text-xs">🔊</button>
        <input type="range" min={0} max={1} step={0.05} value={volume} onChange={handleVolume}
          className="w-20 accent-blue-500 cursor-pointer" />
        <button onClick={toggleFS} className="text-white/80 hover:text-white transition-colors">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
          </svg>
        </button>
      </div>
    </div>
  )
}

function MiniStat({ icon, label, value, color }) {
  return (
    <div className="rounded-xl p-3 text-center" style={{ background: 'var(--color-surface-2)', borderLeft: `2px solid ${color}` }}>
      <div className="text-xl mb-1">{icon}</div>
      <div className="font-mono font-bold text-text-primary text-sm">{value}</div>
      <div className="text-xs text-text-secondary mt-0.5">{label}</div>
    </div>
  )
}

export default function VideoPlayerPage() {
  const navigate = useNavigate()
  const { jobId, results } = useAnalysis()
  const videoRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.8)
  const [showControls, setShowControls] = useState(true)
  const [activeEvent, setActiveEvent] = useState(null)
  const [videoError, setVideoError] = useState(null)
  const timerRef = useRef(null)

  const timeline = results?.timeline || results?.events || []
  const fps = results?.fps || 24
  const t1Stats = results?.teams?.[0]?.metrics || results?.teams?.[0] || {}
  const t2Stats = results?.teams?.[1]?.metrics || results?.teams?.[1] || {}

  function handleMouseMove() {
    setShowControls(true)
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setShowControls(false), 3000)
  }
  useEffect(() => () => clearTimeout(timerRef.current), [])

  if (!jobId) return (
    <div className="p-8 flex flex-col items-center justify-center min-h-screen gap-6">
      <div className="card max-w-md w-full text-center space-y-4">
        <span className="text-5xl">🎬</span>
        <h2 className="text-xl font-bold text-text-primary">Chưa có video</h2>
        <p className="text-text-secondary text-sm">Hãy tải lên và phân tích một video trận đấu trước.</p>
        <button onClick={() => navigate('/upload')}
          className="px-5 py-2.5 rounded-lg font-medium text-sm text-white" style={{ background: 'var(--color-team-1)' }}>
          Tải lên video
        </button>
      </div>
    </div>
  )

  return (
    <div className="p-8 space-y-6">
      <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Video Player</h1>
          <p className="text-text-secondary mt-1 text-sm">Xem lại video với phân tích trực quan</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <Badge variant="accent" size="md">Job: {jobId?.slice(0, 8)}…</Badge>
          {results?.input_filename && (
            <p className="text-xs text-text-secondary font-mono max-w-xs truncate" title={results.input_md5}>
              {results.input_filename}
              {results.input_size_bytes ? ` · ${(results.input_size_bytes / 1048576).toFixed(1)} MB` : ''}
            </p>
          )}
        </div>
      </motion.div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Video */}
        <div className="xl:col-span-2 space-y-4">
          <div className="relative rounded-xl overflow-hidden bg-black" style={{ aspectRatio: '16/9' }}
            onMouseMove={handleMouseMove}>
            {!videoError ? (
            <video key={jobId} ref={videoRef} src={getVideoUrl(jobId)} className="w-full h-full object-contain"
              onTimeUpdate={e => setCurrentTime(e.target.currentTime)}
              onLoadedMetadata={e => { setDuration(e.target.duration); setVideoError(null) }}
              onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)}
              onError={() => setVideoError('Không phát được video. Chạy lại phân tích hoặc cài ffmpeg để tạo MP4.')} />
            ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center">
              <p className="text-red-400 text-sm">{videoError}</p>
              <p className="text-text-secondary text-xs">Job: {jobId?.slice(0, 8)}… — thử upload lại nếu vừa restart backend.</p>
              <button type="button" onClick={() => { setVideoError(null); videoRef.current?.load() }}
                className="px-4 py-2 rounded-lg text-sm text-white" style={{ background: 'var(--color-team-1)' }}>
                Thử tải lại
              </button>
            </div>
            )}
            <motion.div animate={{ opacity: showControls || !playing ? 1 : 0 }} transition={{ duration: 0.3 }}>
              <VideoControls videoRef={videoRef} playing={playing} setPlaying={setPlaying}
                duration={duration} currentTime={currentTime} volume={volume} setVolume={setVolume} />
            </motion.div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MiniStat icon="⚽" label="Kiểm soát bóng" value={`${(t1Stats.possession || 50).toFixed(0)}%`} color="var(--color-team-1)" />
            <MiniStat icon="⚡" label="Tốc độ TB" value={`${(t1Stats.avg_speed || 0).toFixed(1)} km/h`} color="var(--color-accent)" />
            <MiniStat icon="🔥" label="Pressing" value={(t1Stats.pressing_intensity || 0).toFixed(1)} color="var(--color-team-2)" />
            <MiniStat icon="🧩" label="Đội hình" value={`${(t1Stats.formation_adherence || 0).toFixed(0)}%`} color="var(--color-grade-c)" />
          </div>
        </div>

        {/* Info panel */}
        <div className="xl:col-span-1 space-y-4 max-h-screen overflow-y-auto">
          <Card>
            <h3 className="font-bold text-text-primary text-sm mb-3 flex items-center gap-2">
              <span className="w-1 h-4 rounded" style={{ background: 'var(--color-team-1)' }} /> Sự kiện trận đấu
            </h3>
            {timeline.length > 0 ? (
              <div className="space-y-1 max-h-80 overflow-y-auto pr-1">
                {timeline.map((event, i) => (
                  <button key={i} onClick={() => { if (videoRef.current && event.frame != null) videoRef.current.currentTime = event.frame / fps; setActiveEvent(event) }}
                    className={`w-full text-left flex items-start gap-3 p-3 rounded-lg transition-all ${
                      activeEvent === event ? 'border border-blue-500/40 bg-blue-500/10' : 'border border-transparent hover:bg-surface-2'}`}>
                    <span className="text-base flex-shrink-0 mt-0.5">
                      {{ goal: '⚽', card: '🟨', foul: '⚠️', shot: '🎯', corner: '🚩' }[event.type] || '📍'}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs text-blue-400 font-semibold">{formatTime((event.frame || 0) / fps)}</span>
                        <Badge variant={event.team === 0 ? 'team1' : 'team2'} size="sm">Đội {(event.team ?? 0) + 1}</Badge>
                      </div>
                      <p className="text-xs text-text-secondary mt-0.5 leading-relaxed">{event.description || event.type}</p>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-text-secondary text-sm">
                <span className="text-2xl block mb-2">📭</span>Không có dữ liệu sự kiện
              </div>
            )}
          </Card>

          <Card>
            <h3 className="font-bold text-text-primary text-sm mb-4 flex items-center gap-2">
              <span className="w-1 h-4 rounded" style={{ background: 'var(--color-accent)' }} /> Thống kê nhanh
            </h3>
            <div className="space-y-3">
              {[
                { label: 'Kiểm soát bóng', t1: t1Stats.possession, t2: t2Stats.possession, unit: '%' },
                { label: 'Tốc độ TB (km/h)', t1: t1Stats.avg_speed, t2: t2Stats.avg_speed, unit: '' },
                { label: 'Pressing', t1: t1Stats.pressing_intensity, t2: t2Stats.pressing_intensity, unit: '' },
                { label: 'Formation', t1: t1Stats.formation_adherence, t2: t2Stats.formation_adherence, unit: '%' },
              ].map(stat => {
                const v1 = parseFloat(stat.t1) || 0, v2 = parseFloat(stat.t2) || 0, total = v1 + v2 || 1
                return (
                  <div key={stat.label}>
                    <div className="flex justify-between text-xs text-text-secondary mb-1">
                      <span className="text-blue-400 font-mono font-medium">{v1.toFixed(1)}{stat.unit}</span>
                      <span>{stat.label}</span>
                      <span className="text-red-400 font-mono font-medium">{v2.toFixed(1)}{stat.unit}</span>
                    </div>
                    <div className="h-1.5 rounded-full overflow-hidden flex" style={{ background: 'var(--color-surface-2)' }}>
                      <div className="h-full" style={{ width: `${(v1 / total) * 100}%`, background: 'var(--color-team-1)' }} />
                      <div className="h-full" style={{ width: `${(v2 / total) * 100}%`, background: 'var(--color-team-2)' }} />
                    </div>
                  </div>
                )
              })}
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

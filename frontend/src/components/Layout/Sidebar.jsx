import { NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAnalysis } from '../../context/AnalysisContext'

const NAV_ITEMS = [
  { to: '/upload', label: 'Tải lên', icon: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
    </svg>
  )},
  { to: '/dashboard', label: 'Phân tích', icon: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5" />
    </svg>
  )},
  { to: '/report', label: 'Báo cáo', icon: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  )},
  { to: '/video', label: 'Video', icon: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
      <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z" />
    </svg>
  )},
]

const statusConfig = {
  idle:       { label: 'Chờ video',  color: 'bg-gray-500',   pulse: false },
  uploading:  { label: 'Đang tải',   color: 'bg-yellow-500', pulse: true },
  processing: { label: 'Đang xử lý', color: 'bg-blue-500',   pulse: true },
  done:       { label: 'Hoàn thành', color: 'bg-emerald-500',pulse: false },
  error:      { label: 'Lỗi',        color: 'bg-red-500',    pulse: false },
}

export default function Sidebar() {
  const { status, processingProgress } = useAnalysis()
  const cfg = statusConfig[status] || statusConfig.idle

  return (
    <aside className="no-print fixed left-0 top-0 h-screen w-64 flex flex-col z-30"
      style={{ background: 'var(--color-surface)', borderRight: '1px solid var(--color-border)' }}>

      {/* Logo */}
      <div className="px-6 py-6 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
            style={{ background: 'linear-gradient(135deg, var(--color-team-1), var(--color-accent))' }}>
            <span className="text-lg">⚽</span>
          </div>
          <div>
            <div className="text-sm font-bold text-text-primary leading-tight">Football</div>
            <div className="text-xs font-semibold leading-tight" style={{ color: 'var(--color-team-1)' }}>Analytics AI</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => (
          <NavLink key={item.to} to={item.to}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            {item.icon}
            <span className="text-sm">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Status */}
      <div className="px-4 py-4 border-t border-border">
        <div className="rounded-lg p-3" style={{ background: 'var(--color-surface-2)' }}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">Trạng thái</span>
            <div className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${cfg.color} ${cfg.pulse ? 'animate-pulse' : ''}`} />
              <span className="text-xs font-medium text-text-primary">{cfg.label}</span>
            </div>
          </div>
          {status === 'processing' && (
            <div>
              <div className="metric-bar-track">
                <motion.div className="metric-bar-fill"
                  style={{ background: 'var(--color-team-1)', width: `${processingProgress}%` }}
                  animate={{ width: `${processingProgress}%` }}
                  transition={{ duration: 0.5 }} />
              </div>
              <div className="text-right mt-1">
                <span className="text-xs font-mono text-text-secondary">{processingProgress}%</span>
              </div>
            </div>
          )}
          {status === 'done' && (
            <p className="text-xs mt-1" style={{ color: 'var(--color-accent)' }}>Sẵn sàng xem kết quả</p>
          )}
        </div>
      </div>
    </aside>
  )
}

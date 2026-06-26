const gradeConfig = {
  A: { label: 'A', bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/40', ring: '#10B981' },
  B: { label: 'B', bg: 'bg-blue-500/20',    text: 'text-blue-400',    border: 'border-blue-500/40',    ring: '#3B82F6' },
  C: { label: 'C', bg: 'bg-amber-500/20',   text: 'text-amber-400',   border: 'border-amber-500/40',   ring: '#F59E0B' },
  D: { label: 'D', bg: 'bg-orange-500/20',  text: 'text-orange-400',  border: 'border-orange-500/40',  ring: '#F97316' },
  F: { label: 'F', bg: 'bg-red-500/20',     text: 'text-red-400',     border: 'border-red-500/40',     ring: '#EF4444' },
}

export default function GradeChip({ grade = 'B', size = 'md', showLabel = false, className = '' }) {
  const cfg = gradeConfig[grade?.toUpperCase()] || gradeConfig.B
  const sizes = { sm: 'text-sm w-8 h-8', md: 'text-base w-10 h-10', lg: 'text-xl w-14 h-14', xl: 'text-3xl w-20 h-20' }
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <span className={`inline-flex items-center justify-center rounded-xl font-bold border ${cfg.bg} ${cfg.text} ${cfg.border} ${sizes[size]}`}
        style={{ boxShadow: `0 0 12px ${cfg.ring}30` }}>
        {cfg.label}
      </span>
      {showLabel && (
        <div className="flex flex-col items-start leading-tight">
          <span className={`text-sm font-semibold ${cfg.text}`}>
            Hạng {cfg.label} — {grade === 'A' ? 'Xuất sắc' : grade === 'B' ? 'Tốt' : grade === 'C' ? 'Trung bình' : grade === 'D' ? 'Yếu' : 'Kém'}
          </span>
          <span className="text-[10px] text-text-secondary">theo thang A → F</span>
        </div>
      )}
    </div>
  )
}

export { gradeConfig }

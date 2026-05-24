import { motion } from 'framer-motion'

export default function MetricBar({ label, team1Value, team2Value, unit = '', format = (v) => v, icon }) {
  const t1 = parseFloat(team1Value) || 0
  const t2 = parseFloat(team2Value) || 0
  const total = t1 + t2 || 1

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5 text-xs">
        {icon && <span className="opacity-70">{icon}</span>}
        <span className="text-text-secondary font-medium">{label}</span>
      </div>
      <div className="flex items-center gap-3">
        <span className="font-mono text-sm font-semibold text-blue-400 w-16 text-right">
          {format(t1)}{unit}
        </span>
        <div className="flex-1 flex h-2 rounded-full overflow-hidden bg-surface-2">
          <motion.div className="h-full rounded-l-full"
            style={{ background: 'var(--color-team-1)' }}
            initial={{ width: 0 }} animate={{ width: `${(t1 / total) * 100}%` }}
            transition={{ duration: 0.8, delay: 0.1 }} />
          <motion.div className="h-full rounded-r-full"
            style={{ background: 'var(--color-team-2)' }}
            initial={{ width: 0 }} animate={{ width: `${(t2 / total) * 100}%` }}
            transition={{ duration: 0.8, delay: 0.1 }} />
        </div>
        <span className="font-mono text-sm font-semibold text-red-400 w-16 text-left">
          {format(t2)}{unit}
        </span>
      </div>
    </div>
  )
}

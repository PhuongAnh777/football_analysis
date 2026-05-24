import { motion } from 'framer-motion'

const SIZE   = { sm: 72,  md: 96,  lg: 120, xl: 160 }
const STROKE = { sm: 4,   md: 5,   lg: 6,   xl: 8   }

export default function ProgressRing({ value = 0, color, size = 'md', children, className = '' }) {
  const px = SIZE[size] || 96
  const stroke = STROKE[size] || 5
  const r = (px - stroke * 2) / 2
  const circumference = 2 * Math.PI * r
  const offset = circumference - (value / 100) * circumference

  return (
    <div className={`relative inline-flex items-center justify-center ${className}`}
      style={{ width: px, height: px }}>
      <svg width={px} height={px} viewBox={`0 0 ${px} ${px}`} className="absolute inset-0">
        <circle cx={px/2} cy={px/2} r={r} fill="none"
          stroke="var(--color-surface-2)" strokeWidth={stroke} />
        <motion.circle cx={px/2} cy={px/2} r={r} fill="none"
          stroke={color || 'var(--color-team-1)'} strokeWidth={stroke}
          strokeLinecap="round" strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.2, ease: 'easeOut' }}
          style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }} />
      </svg>
      <div className="relative z-10 flex flex-col items-center justify-center">{children}</div>
    </div>
  )
}

import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import GradeChip from '../UI/GradeChip'

const COLUMNS = [
  { key: 'track_id',            label: '# ID',     sortable: true  },
  { key: 'position',            label: 'Vị trí',   sortable: false },
  { key: 'total_score',         label: 'Điểm',     sortable: true  },
  { key: 'grade',               label: 'Grade',    sortable: false },
  { key: 'avg_speed',           label: 'Tốc độ',   sortable: true  },
  { key: 'pressing',            label: 'Pressing', sortable: true  },
  { key: 'discipline',          label: 'Kỷ luật',  sortable: true  },
  { key: 'coverage',            label: 'Phủ sóng', sortable: true  },
  { key: 'high_intensity_runs', label: 'HI Runs',  sortable: true  },
  { key: 'creative_passes',     label: 'Kiến tạo', sortable: true  },
]

const MEDAL = { 0: '🥇', 1: '🥈', 2: '🥉' }

export default function PlayerTable({ team1Players = [], team2Players = [] }) {
  const [activeTeam, setActiveTeam] = useState(0)
  const [sortKey, setSortKey] = useState('total_score')
  const [sortDir, setSortDir] = useState('desc')

  const players = activeTeam === 0 ? team1Players : team2Players

  const sorted = useMemo(() => {
    return [...players].sort((a, b) => {
      const va = a[sortKey] ?? 0, vb = b[sortKey] ?? 0
      if (typeof va === 'string') return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va)
      return sortDir === 'asc' ? va - vb : vb - va
    })
  }, [players, sortKey, sortDir])

  function handleSort(key) {
    if (key === sortKey) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  return (
    <div>
      <div className="flex gap-2 mb-4">
        {['Đội 1', 'Đội 2'].map((label, i) => (
          <button key={i} onClick={() => setActiveTeam(i)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTeam === i
                ? i === 0 ? 'bg-blue-500/20 text-blue-400 border border-blue-500/40'
                          : 'bg-red-500/20 text-red-400 border border-red-500/40'
                : 'text-text-secondary border border-border hover:border-border-hover'
            }`}>
            {label} <span className="ml-1.5 text-xs opacity-60">({(i === 0 ? team1Players : team2Players).length})</span>
          </button>
        ))}
      </div>

      <div className="overflow-x-auto rounded-xl border border-border">
        <table className="data-table w-full">
          <thead>
            <tr style={{ background: 'var(--color-surface-2)' }}>
              <th className="pl-4">#</th>
              {COLUMNS.map(col => (
                <th key={col.key} onClick={() => col.sortable && handleSort(col.key)}>
                  {col.label}
                  {col.sortable && <span className="ml-1 opacity-60">{sortKey === col.key ? (sortDir === 'asc' ? '↑' : '↓') : '↕'}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <AnimatePresence mode="wait">
              {sorted.length === 0 ? (
                <tr key="empty">
                  <td colSpan={COLUMNS.length + 1} className="text-center py-10 text-text-secondary text-sm">
                    Không có dữ liệu cầu thủ
                  </td>
                </tr>
              ) : sorted.map((player, idx) => (
                <motion.tr key={player.track_id ?? idx}
                  initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.03 }}
                  style={idx < 3 ? {
                    background: idx === 0 ? 'rgba(251,191,36,0.04)' : idx === 1 ? 'rgba(156,163,175,0.04)' : 'rgba(180,100,50,0.04)'
                  } : {}}>
                  <td className="pl-4">
                    {idx < 3 ? <span className="text-base">{MEDAL[idx]}</span>
                              : <span className="text-text-secondary text-xs font-mono">{idx + 1}</span>}
                  </td>
                  {COLUMNS.map(col => (
                    <td key={col.key}>
                      {col.key === 'grade' ? <GradeChip grade={player.grade} size="sm" />
                      : col.key === 'total_score' ? <span className="font-mono font-bold text-sm">{Number(player.total_score ?? 0).toFixed(1)}</span>
                      : col.key === 'avg_speed' ? <span className="font-mono text-xs text-text-secondary">{Number(player[col.key] ?? 0).toFixed(1)} km/h</span>
                      : <span className="font-mono text-xs text-text-secondary">
                          {typeof player[col.key] === 'number' ? Number(player[col.key]).toFixed(1) : player[col.key] ?? '—'}
                        </span>}
                    </td>
                  ))}
                </motion.tr>
              ))}
            </AnimatePresence>
          </tbody>
        </table>
      </div>
    </div>
  )
}

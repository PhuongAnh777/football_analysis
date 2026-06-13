import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import GradeChip from '../UI/GradeChip'

const COLUMNS = [
  { key: 'position',            label: 'Vị trí',   sortable: false },
  { key: 'total_score',         label: 'Điểm',     sortable: true  },
  { key: 'grade',               label: 'Xếp loại', sortable: false },
  { key: 'avg_speed',           label: 'Tốc độ',   sortable: true  },
  { key: 'pressing',            label: 'Pressing', sortable: true  },
  { key: 'discipline',          label: 'Kỷ luật',  sortable: true  },
  { key: 'coverage',            label: 'Phủ sóng', sortable: true  },
  { key: 'high_intensity_runs', label: 'HI Runs',  sortable: true  },
  { key: 'creative_passes',     label: 'Kiến tạo', sortable: true  },
]

const MEDAL = { 0: '🥇', 1: '🥈', 2: '🥉' }

function TeamPlayerTable({ players, teamLabel, accentClass }) {
  const [sortKey, setSortKey] = useState('total_score')
  const [sortDir, setSortDir] = useState('desc')

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
    <div className="min-w-0">
      <p className={`text-sm font-semibold mb-3 ${accentClass}`}>
        {teamLabel} <span className="text-xs opacity-60 font-normal">({players.length} cầu thủ)</span>
      </p>
      <div className="overflow-x-auto rounded-xl border border-border">
        <table className="data-table w-full">
          <thead>
            <tr style={{ background: 'var(--color-surface-2)' }}>
              <th className="pl-4">Hạng</th>
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
                <motion.tr key={player.track_id ?? `${teamLabel}-${idx}`}
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
                      : col.key === 'position' ? <span className="text-xs text-text-primary">{player.position || '—'}</span>
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

export default function PlayerTable({ team1Players = [], team2Players = [], team1Name = 'Đội 1', team2Name = 'Đội 2' }) {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
      <TeamPlayerTable players={team1Players} teamLabel={team1Name} accentClass="text-blue-400" />
      <TeamPlayerTable players={team2Players} teamLabel={team2Name} accentClass="text-red-400" />
    </div>
  )
}

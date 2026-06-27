import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const COLUMNS = [
  { key: 'track_id',          label: 'ID',                    sortable: true  },
  { key: 'pass_success_pct',  label: 'Chuyền thành công %',   sortable: true  },
  { key: 'total_passes',      label: 'Tổng đường chuyền',     sortable: true  },
  { key: 'key_passes',        label: 'Chuyền tạo cơ hội',     sortable: true  },
  { key: 'avg_speed',         label: 'Tốc độ (km/h)',         sortable: true  },
  { key: 'pressing',          label: 'Pressing %',            sortable: true  },
]

function formatCell(col, player) {
  const val = player[col.key]
  if (col.key === 'track_id') {
    return <span className="font-mono text-xs text-text-primary">#{val ?? '—'}</span>
  }
  if (col.key === 'pass_success_pct') {
    return val != null
      ? <span className="font-mono text-xs text-text-secondary">{Number(val).toFixed(1)}%</span>
      : <span className="text-xs text-text-secondary">—</span>
  }
  if (col.key === 'avg_speed') {
    return <span className="font-mono text-xs text-text-secondary">{Number(val ?? 0).toFixed(1)} km/h</span>
  }
  if (col.key === 'pressing') {
    return <span className="font-mono text-xs text-text-secondary">{Number(val ?? 0).toFixed(1)}%</span>
  }
  return (
    <span className="font-mono text-xs text-text-secondary">
      {typeof val === 'number' ? val : val ?? '—'}
    </span>
  )
}

function TeamPlayerTable({ players, teamLabel, accentClass }) {
  const [sortKey, setSortKey] = useState('key_passes')
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
              {COLUMNS.map(col => (
                <th key={col.key} className={col.key === 'track_id' ? 'pl-4' : ''} onClick={() => col.sortable && handleSort(col.key)}>
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
                  <td colSpan={COLUMNS.length} className="text-center py-10 text-text-secondary text-sm">
                    Không có dữ liệu cầu thủ
                  </td>
                </tr>
              ) : sorted.map((player, idx) => (
                <motion.tr key={player.track_id ?? `${teamLabel}-${idx}`}
                  initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.03 }}>
                  {COLUMNS.map(col => (
                    <td key={col.key} className={col.key === 'track_id' ? 'pl-4' : ''}>
                      {formatCell(col, player)}
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

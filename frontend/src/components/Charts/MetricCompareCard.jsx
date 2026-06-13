import MetricBar from '../UI/MetricBar'

const METRICS = [
  { key: 'possession',            label: 'Kiểm soát bóng',       unit: '%',     icon: '⚽' },
  { key: 'compact_score',         label: 'Độ compact đội hình',  unit: ' m',    icon: '📐' },
  { key: 'pressing_intensity',    label: 'Cường độ pressing',    unit: '',      icon: '🔥' },
  { key: 'formation_adherence',   label: 'Tuân thủ đội hình',   unit: '%',     icon: '🧩' },
  { key: 'avg_speed',             label: 'Tốc độ trung bình',    unit: ' km/h', icon: '⚡' },
  { key: 'sprint_pct',            label: 'Tỷ lệ sprint',         unit: '%',     icon: '🏃' },
  { key: 'defensive_line_height', label: 'Độ cao hàng thủ',      unit: ' m',    icon: '🛡️' },
  { key: 'width',                 label: 'Độ rộng đội hình',     unit: ' m',    icon: '↔️' },
  { key: 'high_intensity_runs',   label: 'Chạy cường độ cao',    unit: '',      icon: '💨' },
  { key: 'ball_recoveries',       label: 'Thu hồi bóng',         unit: '',      icon: '🔄' },
  { key: 'dangerous_turnovers',   label: 'Mất bóng nguy hiểm',   unit: '',      icon: '⚠️' },
  { key: 'forward_passes_pct',    label: 'Chuyền tiến công',     unit: '%',     icon: '➡️' },
]

export default function MetricCompareCard({ team1Stats, team2Stats, team1Name = 'Đội 1', team2Name = 'Đội 2' }) {
  const half = Math.ceil(METRICS.length / 2)
  const columns = [METRICS.slice(0, half), METRICS.slice(half)]

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-x-8 gap-y-4">
      {columns.map((group, colIdx) => (
        <div key={colIdx} className="space-y-4">
          {group.map((m) => (
            <MetricBar key={m.key} label={m.label} icon={m.icon}
              team1Value={team1Stats?.[m.key] ?? 0}
              team2Value={team2Stats?.[m.key] ?? 0}
              team1Name={team1Name}
              team2Name={team2Name}
              unit={m.unit}
              format={(v) => Number(v).toFixed(1)} />
          ))}
        </div>
      ))}
    </div>
  )
}

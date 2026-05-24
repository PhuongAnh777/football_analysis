import MetricBar from '../UI/MetricBar'

const METRICS = [
  { key: 'possession',            label: 'Kiểm soát bóng',    unit: '%',     icon: '⚽' },
  { key: 'compact_score',         label: 'Compact score',      unit: ' m',    icon: '📐' },
  { key: 'pressing_intensity',    label: 'Pressing intensity', unit: '',      icon: '🔥' },
  { key: 'formation_adherence',   label: 'Formation adherence',unit: '%',     icon: '🧩' },
  { key: 'avg_speed',             label: 'Tốc độ TB',          unit: ' km/h', icon: '⚡' },
  { key: 'sprint_pct',            label: 'Sprint %',           unit: '%',     icon: '🏃' },
  { key: 'defensive_line_height', label: 'Độ cao hàng thủ',    unit: ' m',    icon: '🛡️' },
  { key: 'width',                 label: 'Độ rộng đội hình',   unit: ' m',    icon: '↔️' },
  { key: 'high_intensity_runs',   label: 'High intensity runs', unit: '',     icon: '💨' },
  { key: 'ball_recoveries',       label: 'Ball recoveries',    unit: '',      icon: '🔄' },
  { key: 'dangerous_turnovers',   label: 'Turnovers nguy hiểm',unit: '',      icon: '⚠️' },
  { key: 'forward_passes_pct',    label: 'Pass tiến công',     unit: '%',     icon: '➡️' },
]

export default function MetricCompareCard({ team1Stats, team2Stats }) {
  return (
    <div className="grid grid-cols-1 gap-4">
      {METRICS.map((m) => (
        <MetricBar key={m.key} label={m.label} icon={m.icon}
          team1Value={team1Stats?.[m.key] ?? 0}
          team2Value={team2Stats?.[m.key] ?? 0}
          unit={m.unit}
          format={(v) => Number(v).toFixed(1)} />
      ))}
    </div>
  )
}

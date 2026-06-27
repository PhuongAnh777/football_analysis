import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Legend, Tooltip,
} from 'recharts'

const AXES = [
  { key: 'kiem_soat_bong', label: 'Kiểm soát bóng' },
  { key: 'pressing',       label: 'Pressing' },
  { key: 'toc_do',         label: 'Tốc độ trung bình' },
  { key: 'do_compact',     label: 'Độ compact đội hình' },
]

function buildData(t1, t2) {
  return AXES.map(({ key, label }) => ({
    axis: label,
    team1: Number(t1?.[key] ?? 0),
    team2: Number(t2?.[key] ?? 0),
  }))
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="card py-2 px-3 text-xs space-y-1 shadow-xl">
      <p className="font-semibold text-text-primary mb-1">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-text-secondary">{p.name}:</span>
          <span className="font-mono font-semibold" style={{ color: p.color }}>{Number(p.value).toFixed(1)}</span>
        </div>
      ))}
    </div>
  )
}

export default function RadarComparison({ team1Name = 'Đội 1', team2Name = 'Đội 2', team1Scores, team2Scores }) {
  const data = buildData(team1Scores, team2Scores)
  return (
    <ResponsiveContainer width="100%" height={360}>
      <RadarChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
        <PolarGrid stroke="var(--color-border)" strokeOpacity={0.6} />
        <PolarAngleAxis dataKey="axis"
          tick={{ fill: 'var(--color-text-secondary)', fontSize: 11, fontFamily: 'Inter' }} />
        <PolarRadiusAxis angle={90} domain={[0, 100]}
          tick={{ fill: 'var(--color-text-secondary)', fontSize: 9 }}
          tickCount={5} stroke="var(--color-border)" strokeOpacity={0.4} />
        <Radar name={team1Name} dataKey="team1"
          stroke="var(--color-team-1)" fill="var(--color-team-1)" fillOpacity={0.2}
          strokeWidth={2} dot={{ r: 3, fill: 'var(--color-team-1)' }} />
        <Radar name={team2Name} dataKey="team2"
          stroke="var(--color-team-2)" fill="var(--color-team-2)" fillOpacity={0.15}
          strokeWidth={2} dot={{ r: 3, fill: 'var(--color-team-2)' }} />
        <Legend wrapperStyle={{ fontSize: 12, color: 'var(--color-text-secondary)', paddingTop: 12 }} iconType="circle" iconSize={8} />
        <Tooltip content={<CustomTooltip />} />
      </RadarChart>
    </ResponsiveContainer>
  )
}

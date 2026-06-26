import Badge from './Badge'

export default function TeamHeader({ teamName, formation, profile, possession, side = 'team1' }) {
  const isTeam1 = side === 'team1'
  const accentClass = isTeam1 ? 'text-blue-400' : 'text-red-400'
  const borderClass = isTeam1 ? 'border-blue-500/30 bg-blue-500/5' : 'border-red-500/30 bg-red-500/5'

  return (
    <div className={`rounded-xl border p-6 ${borderClass} flex flex-col items-center text-center gap-4`}>
      <div className="flex flex-col items-center gap-1">
        <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide">Kiểm soát bóng</p>
        <span className={`text-4xl font-bold font-mono ${accentClass}`}>
          {Number(possession || 0).toFixed(1)}%
        </span>
      </div>
      <div>
        <h3 className="text-xl font-bold text-text-primary">{teamName}</h3>
        <div className="flex flex-wrap items-center justify-center gap-2 mt-2">
          <Badge variant={isTeam1 ? 'team1' : 'team2'} size="md">{formation}</Badge>
          {profile && <Badge variant="ghost" size="md">{profile}</Badge>}
        </div>
      </div>
    </div>
  )
}

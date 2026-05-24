import GradeChip from './GradeChip'
import ProgressRing from './ProgressRing'
import Badge from './Badge'

export default function TeamHeader({ teamName, formation, profile, score, grade, side = 'team1' }) {
  const isTeam1 = side === 'team1'
  const ringColor = isTeam1 ? 'var(--color-team-1)' : 'var(--color-team-2)'
  const accentClass = isTeam1 ? 'text-blue-400' : 'text-red-400'
  const borderClass = isTeam1 ? 'border-blue-500/30 bg-blue-500/5' : 'border-red-500/30 bg-red-500/5'

  return (
    <div className={`rounded-xl border p-6 ${borderClass} flex flex-col items-center text-center gap-4`}>
      <ProgressRing value={score} color={ringColor} size="xl">
        <span className={`text-3xl font-bold font-mono ${accentClass}`}>{score}</span>
        <span className="text-xs text-text-secondary">/ 100</span>
      </ProgressRing>
      <div>
        <h3 className="text-xl font-bold text-text-primary">{teamName}</h3>
        <div className="flex flex-wrap items-center justify-center gap-2 mt-2">
          <Badge variant={isTeam1 ? 'team1' : 'team2'} size="md">{formation}</Badge>
          <Badge variant="ghost" size="md">{profile}</Badge>
        </div>
      </div>
      <GradeChip grade={grade} size="lg" showLabel />
    </div>
  )
}

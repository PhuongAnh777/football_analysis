import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAnalysis } from '../context/AnalysisContext'
import { getResults } from '../api/client'
import Card from '../components/UI/Card'
import Badge from '../components/UI/Badge'
import TeamHeader from '../components/UI/TeamHeader'
import RadarComparison from '../components/Charts/RadarComparison'
import MetricCompareCard from '../components/Charts/MetricCompareCard'
import PlayerTable from '../components/Charts/PlayerTable'

function Skeleton({ className = '' }) {
  return <div className={`rounded-lg animate-pulse ${className}`} style={{ background: 'var(--color-surface-2)' }} />
}

function DualTeamImage({ team1Image, team2Image, team1Label = 'Đội 1', team2Label = 'Đội 2', alt }) {
  const teams = [
    { image: team1Image, label: team1Label, color: 'text-blue-400', border: 'border-blue-500/30' },
    { image: team2Image, label: team2Label, color: 'text-red-400', border: 'border-red-500/30' },
  ]
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      {teams.map((team, i) => (
        <div key={i} className="space-y-2">
          <p className={`text-xs font-semibold ${team.color}`}>{team.label}</p>
          <div className={`rounded-xl overflow-hidden border ${team.border}`}
            style={{ background: 'var(--color-surface-2)', minHeight: 200 }}>
            {team.image
              ? <img src={`data:image/png;base64,${team.image}`} alt={`${alt} — ${team.label}`} className="w-full h-auto object-contain" />
              : <div className="flex items-center justify-center h-48 text-text-secondary text-sm">Không có dữ liệu</div>}
          </div>
        </div>
      ))}
    </div>
  )
}

const container = { hidden: {}, show: { transition: { staggerChildren: 0.07 } } }
const item = { hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0, transition: { duration: 0.35 } } }

function extractTeamData(results, idx) {
  return results?.teams?.[idx] || null
}
function extractEvaluation(results) {
  return results?.evaluation || results?.match_report?.evaluation || null
}
function extractPlayers(results, idx) {
  const teamNum = idx + 1
  const all = results?.players || results?.match_report?.player_report || []
  if (Array.isArray(all)) {
    return all.filter(p => p.team === teamNum || p.team_id === teamNum)
  }
  return []
}
function extractCharts(results) { return results?.charts || {} }
function buildRadarScores(teamData) {
  if (!teamData) return {}
  const m = teamData.metrics || teamData
  return {
    kiem_soat_bong: m.possession || m.ball_control || 60,
    doi_hinh:       m.formation_adherence || 65,
    pressing:       m.pressing_intensity || 70,
    ky_luat:        m.discipline || 68,
    toc_do:         m.avg_speed_normalized || (m.avg_speed || 20) / 0.3,
    on_dinh:        m.stability || 65,
    phong_thu:      m.defensive_score || 62,
    do_rong:        m.width_normalized || (m.width || 30) / 0.55,
  }
}

export default function DashboardPage() {
  const navigate = useNavigate()
  const { jobId, results, setResults } = useAnalysis()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!jobId) return
    setLoading(true)
    setError(null)
    setResults(null)
    getResults(jobId).then(setResults).catch(e => setError(e.message)).finally(() => setLoading(false))
  }, [jobId, setResults])

  const evaluation = extractEvaluation(results)
  const team1 = extractTeamData(results, 0)
  const team2 = extractTeamData(results, 1)
  const charts = extractCharts(results)

  const t1Name  = team1?.name || evaluation?.doi_1?.ten || evaluation?.danh_gia_doi?.doi_1?.ten || 'Đội 1'
  const t2Name  = team2?.name || evaluation?.doi_2?.ten || evaluation?.danh_gia_doi?.doi_2?.ten || 'Đội 2'
  const t1Score = Number(team1?.overall_score || evaluation?.doi_1?.diem_tong || evaluation?.danh_gia_doi?.doi_1?.diem_so_tong || 72)
  const t2Score = Number(team2?.overall_score || evaluation?.doi_2?.diem_tong || evaluation?.danh_gia_doi?.doi_2?.diem_so_tong || 68)
  const t1Grade = team1?.grade || evaluation?.doi_1?.xep_loai || evaluation?.danh_gia_doi?.doi_1?.xep_loai || 'B'
  const t2Grade = team2?.grade || evaluation?.doi_2?.xep_loai || evaluation?.danh_gia_doi?.doi_2?.xep_loai || 'C'
  const t1Form  = team1?.formation || evaluation?.doi_1?.so_do || evaluation?.danh_gia_doi?.doi_1?.so_do || '4-3-3'
  const t2Form  = team2?.formation || evaluation?.doi_2?.so_do || evaluation?.danh_gia_doi?.doi_2?.so_do || '4-4-2'
  const t1Prof  = team1?.tactical_profile || evaluation?.doi_1?.phong_cach || evaluation?.danh_gia_doi?.doi_1?.tactical_profile || 'Pressing mạnh'
  const t2Prof  = team2?.tactical_profile || evaluation?.doi_2?.phong_cach || evaluation?.danh_gia_doi?.doi_2?.tactical_profile || 'Phòng ngự sâu'

  if (loading) return (
    <div className="p-8 space-y-8">
      <Skeleton className="h-10 w-64" />
      <div className="grid grid-cols-2 gap-4"><Skeleton className="h-64" /><Skeleton className="h-64" /></div>
      <Skeleton className="h-80" />
    </div>
  )

  if (error) return (
    <div className="p-8 flex flex-col items-center justify-center min-h-screen gap-4">
      <div className="card max-w-md w-full text-center space-y-4 border-red-500/40">
        <span className="text-4xl">⚠️</span>
        <h2 className="text-xl font-bold text-text-primary">Không thể tải kết quả</h2>
        <p className="text-text-secondary text-sm">{error}</p>
        <div className="flex gap-3 justify-center">
          <button onClick={() => { setError(null); setLoading(true) }}
            className="px-4 py-2 rounded-lg text-sm font-medium text-white" style={{ background: 'var(--color-team-1)' }}>
            Thử lại
          </button>
          <button onClick={() => navigate('/upload')}
            className="px-4 py-2 rounded-lg text-sm font-medium border border-border text-text-secondary hover:text-text-primary">
            Tải lên video mới
          </button>
        </div>
      </div>
    </div>
  )

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }}
        className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Kết quả phân tích trận đấu</h1>
          <p className="text-text-secondary mt-1 text-sm">
            {results?.input_filename || 'Phân tích chiến thuật toàn diện'}
          </p>
        </div>
        <Badge variant="ai" size="lg"><span>✦</span> AI Powered</Badge>
      </motion.div>

      {/* Section 1: Teams */}
      <motion.section variants={container} initial="hidden" animate="show">
        <motion.h2 variants={item} className="text-lg font-bold text-text-primary mb-4 flex items-center gap-2">
          <span className="w-1 h-5 rounded" style={{ background: 'var(--color-team-1)' }} /> Tổng quan trận đấu
        </motion.h2>
        <motion.div variants={item} className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <TeamHeader teamName={t1Name} formation={t1Form} profile={t1Prof} score={t1Score} grade={t1Grade} side="team1" />
          <TeamHeader teamName={t2Name} formation={t2Form} profile={t2Prof} score={t2Score} grade={t2Grade} side="team2" />
        </motion.div>
      </motion.section>

      {/* Section 2: Radar */}
      <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
        <Card>
          <h2 className="text-lg font-bold text-text-primary mb-2 flex items-center gap-2">
            <span className="w-1 h-5 rounded" style={{ background: 'var(--color-accent)' }} /> So sánh chỉ số chiến thuật
          </h2>
          <p className="text-text-secondary text-xs mb-4">Điểm theo 8 chiều chiến thuật</p>
          <RadarComparison team1Name={t1Name} team2Name={t2Name}
            team1Scores={buildRadarScores(team1)} team2Scores={buildRadarScores(team2)} />
        </Card>
      </motion.section>

      {/* Section 3: Metrics */}
      <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <Card>
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-lg font-bold text-text-primary flex items-center gap-2">
              <span className="w-1 h-5 rounded" style={{ background: 'var(--color-team-2)' }} /> Chỉ số so sánh
            </h2>
            <div className="flex items-center gap-4 text-xs">
              <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-blue-400" /><span className="text-text-secondary">{t1Name}</span></div>
              <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-red-400" /><span className="text-text-secondary">{t2Name}</span></div>
            </div>
          </div>
          <MetricCompareCard
            team1Stats={team1?.metrics || team1}
            team2Stats={team2?.metrics || team2}
            team1Name={t1Name}
            team2Name={t2Name}
          />
        </Card>
      </motion.section>

      {/* Section 4: Charts */}
      <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}
        className="space-y-6">
        <Card>
          <h2 className="text-lg font-bold text-text-primary mb-4 flex items-center gap-2">
            <span className="w-1 h-5 rounded" style={{ background: 'var(--color-team-1)' }} /> Heatmap vị trí
          </h2>
          <DualTeamImage
            team1Image={charts.heatmap_team1}
            team2Image={charts.heatmap_team2}
            team1Label={t1Name}
            team2Label={t2Name}
            alt="Heatmap vị trí"
          />
        </Card>
        <Card>
          <h2 className="text-lg font-bold text-text-primary mb-4 flex items-center gap-2">
            <span className="w-1 h-5 rounded" style={{ background: 'var(--color-team-2)' }} /> Mạng lưới chuyền bóng
          </h2>
          <DualTeamImage
            team1Image={charts.passing_network_team1}
            team2Image={charts.passing_network_team2}
            team1Label={t1Name}
            team2Label={t2Name}
            alt="Mạng lưới chuyền bóng"
          />
        </Card>
      </motion.section>

      {/* Section 5: Players */}
      <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
        <Card>
          <h2 className="text-lg font-bold text-text-primary mb-5 flex items-center gap-2">
            <span className="w-1 h-5 rounded" style={{ background: 'var(--color-grade-c)' }} /> Hiệu suất cầu thủ
          </h2>
          <PlayerTable
            team1Players={extractPlayers(results, 0)}
            team2Players={extractPlayers(results, 1)}
            team1Name={t1Name}
            team2Name={t2Name}
          />
        </Card>
      </motion.section>
    </div>
  )
}

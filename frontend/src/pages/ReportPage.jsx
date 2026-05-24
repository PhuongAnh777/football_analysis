import { useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAnalysis } from '../context/AnalysisContext'
import Card from '../components/UI/Card'
import Badge from '../components/UI/Badge'
import GradeChip from '../components/UI/GradeChip'

const CATEGORY_META = [
  { key: 'pressing',    label: 'Pressing',      icon: '🔥' },
  { key: 'doi_hinh',   label: 'Đội hình',      icon: '🧩' },
  { key: 'toc_do',     label: 'Tốc độ',        icon: '⚡' },
  { key: 'hang_thu',   label: 'Hàng thủ',      icon: '🛡️' },
  { key: 'do_rong',    label: 'Độ rộng',       icon: '↔️' },
  { key: 'van_dong',   label: 'Vận động',      icon: '🏃' },
  { key: 'tranh_chap', label: 'Tranh chấp',    icon: '⚔️' },
  { key: 'mat_bong',   label: 'Mất bóng',      icon: '↩️' },
  { key: 'chuyen_bong',label: 'Chuyền bóng',   icon: '🎯', altKeys: ['chuyen', 'nhan_xet_chuyen'] },
]

const HEAD2HEAD_CATS = [
  { key: 'pressing',       label: 'Pressing',             icon: '🔥' },
  { key: 'doi_hinh',       label: 'Đội hình',             icon: '🧩' },
  { key: 'the_luc',        label: 'Thể lực',              icon: '💪' },
  { key: 'kiem_soat_bong', label: 'Kiểm soát bóng',       icon: '⚽' },
  { key: 'phong_ngu',      label: 'Chiến lược phòng ngự', icon: '🛡️' },
  { key: 'su_dung_bien',   label: 'Sử dụng biên',         icon: '📐' },
  { key: 'van_dong',       label: 'Vận động',              icon: '🏃' },
  { key: 'tranh_chap',     label: 'Tranh chấp',            icon: '⚔️' },
  { key: 'mat_bong',       label: 'Mất bóng',              icon: '↩️' },
  { key: 'kien_tao',       label: 'Kiến tạo',              icon: '🎯' },
]

function safeText(obj, ...keys) {
  for (const k of keys) {
    const v = obj?.[k]
    if (v && typeof v === 'string' && v.trim()) return v
  }
  return null
}

function safeList(obj, ...keys) {
  for (const k of keys) {
    const v = obj?.[k]
    if (Array.isArray(v) && v.length) return v
  }
  return []
}

function Bullet({ text, type = 'strength' }) {
  return (
    <li className="flex items-start gap-2 text-sm">
      <span className={`mt-0.5 flex-shrink-0 ${type === 'strength' ? 'text-emerald-400' : 'text-red-400'}`}>
        {type === 'strength' ? '✓' : '✗'}
      </span>
      <span className="text-text-secondary leading-relaxed">{text}</span>
    </li>
  )
}

function CategoryBlock({ cat, evalObj }) {
  const keys = [cat.key, ...(cat.altKeys || []), `${cat.key}_nhan_xet`, `nhan_xet_${cat.key}`]
  const text = safeText(evalObj, ...keys)
  if (!text) return null
  return (
    <div className="flex gap-3 p-3 rounded-lg" style={{ background: 'var(--color-surface-2)' }}>
      <span className="text-base flex-shrink-0 mt-0.5">{cat.icon}</span>
      <div>
        <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1">{cat.label}</p>
        <p className="text-sm text-text-secondary leading-relaxed">{text}</p>
      </div>
    </div>
  )
}

function PlayerCard({ player, type = 'best', side = 'team1' }) {
  if (!player) return null
  const isBest = type === 'best'
  return (
    <div className={`rounded-xl border p-5 space-y-3 ${isBest ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-amber-500/30 bg-amber-500/5'}`}>
      <div className="flex items-center gap-3">
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center font-bold text-lg ${
          side === 'team1' ? 'bg-blue-500/20 text-blue-400' : 'bg-red-500/20 text-red-400'}`}>
          #{player.track_id || '?'}
        </div>
        <div>
          <p className="font-semibold text-text-primary text-sm">{isBest ? '⭐ Xuất sắc nhất' : '📈 Cần cải thiện'}</p>
          <p className="text-xs text-text-secondary">{side === 'team1' ? 'Đội 1' : 'Đội 2'} · {player.position || '—'}</p>
        </div>
        {player.grade && <GradeChip grade={player.grade} size="sm" className="ml-auto" />}
      </div>
      {player.reason && <p className="text-sm text-text-secondary leading-relaxed">{player.reason}</p>}
      {player.recommendation && (
        <p className="text-xs italic text-text-secondary border-t border-border pt-3">💡 {player.recommendation}</p>
      )}
      {player.highlights && (
        <div className="flex flex-wrap gap-1.5 pt-1">
          {(Array.isArray(player.highlights) ? player.highlights : [player.highlights]).map((h, i) => (
            <Badge key={i} variant="accent" size="sm">{h}</Badge>
          ))}
        </div>
      )}
    </div>
  )
}

export default function ReportPage() {
  const navigate = useNavigate()
  const { results, jobId } = useAnalysis()

  const evaluation = results?.evaluation || results?.match_report?.evaluation || {}
  const danhGia = evaluation?.danh_gia_doi || {}
  const doi1 = evaluation?.doi_1 || danhGia?.doi_1 || evaluation?.team_1 || {}
  const doi2 = evaluation?.doi_2 || danhGia?.doi_2 || evaluation?.team_2 || {}
  const tong_quan = evaluation?.tong_quan_tran_dau || evaluation?.overview || {}
  const doi_noi_bat = evaluation?.doi_noi_bat || tong_quan?.doi_noi_bat || {}
  const so_sanh = evaluation?.so_sanh_doi_dau || evaluation?.head_to_head || {}
  const ket_luan = safeText(evaluation, 'ket_luan', 'conclusion') || safeText(tong_quan, 'ket_luan') || ''
  const notable = results?.notable_players || evaluation?.notable_players || {}

  const t1Name = doi1?.ten || 'Đội 1', t2Name = doi2?.ten || 'Đội 2'
  const t1Grade = doi1?.xep_loai || doi1?.grade || 'B'
  const t2Grade = doi2?.xep_loai || doi2?.grade || 'C'
  const t1Score = doi1?.diem_tong || doi1?.score || '—'
  const t2Score = doi2?.diem_tong || doi2?.score || '—'

  if (!results && !jobId) return (
    <div className="p-8 flex flex-col items-center justify-center min-h-screen gap-6">
      <div className="card max-w-md w-full text-center space-y-4">
        <span className="text-5xl">📋</span>
        <h2 className="text-xl font-bold text-text-primary">Chưa có báo cáo</h2>
        <p className="text-text-secondary text-sm">Hãy tải lên và phân tích một video trận đấu trước.</p>
        <button onClick={() => navigate('/upload')}
          className="px-5 py-2.5 rounded-lg font-medium text-sm text-white" style={{ background: 'var(--color-team-1)' }}>
          Tải lên video
        </button>
      </div>
    </div>
  )

  return (
    <div className="p-8 max-w-5xl mx-auto">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }}
        className="flex items-start justify-between flex-wrap gap-4 mb-8 no-print">
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Báo cáo phân tích chiến thuật</h1>
          <p className="text-text-secondary mt-1 text-sm">Phân tích chi tiết bởi Football Analytics AI</p>
        </div>
        <button onClick={() => window.print()}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium border border-border text-text-secondary hover:text-text-primary hover:border-border-hover transition-all">
          🖨️ Xuất PDF
        </button>
      </motion.div>

      <div className="space-y-8">
        {/* Section 1: Tổng quan */}
        <Card animate delay={0}>
          <div className="flex items-center gap-3 mb-5">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center text-sm" style={{ background: 'rgba(59,130,246,0.15)' }}>📊</div>
            <h2 className="text-xl font-bold text-text-primary">Tổng quan trận đấu</h2>
          </div>
          {safeText(tong_quan, 'nhan_xet_chung', 'overview', 'summary') && (
            <blockquote className="relative pl-5 mb-4">
              <span className="absolute left-0 top-0 text-4xl leading-none opacity-30" style={{ color: 'var(--color-team-1)' }}>"</span>
              <p className="text-text-secondary text-sm leading-relaxed italic">
                {safeText(tong_quan, 'nhan_xet_chung', 'overview', 'summary')}
              </p>
            </blockquote>
          )}
          {(doi_noi_bat?.ten || doi_noi_bat?.name) && (
            <div className="flex items-start gap-3 p-4 rounded-xl" style={{ background: 'var(--color-surface-2)', border: '1px solid var(--color-border)' }}>
              <span className="text-2xl">🏆</span>
              <div>
                <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-0.5">Đội nổi bật</p>
                <p className="font-bold text-text-primary">{doi_noi_bat.ten || doi_noi_bat.name}</p>
                {doi_noi_bat.ly_do && <p className="text-sm text-text-secondary mt-1">{doi_noi_bat.ly_do}</p>}
              </div>
            </div>
          )}
        </Card>

        {/* Section 2: Đánh giá từng đội */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {[
            { evalObj: doi1, name: t1Name, grade: t1Grade, score: t1Score, side: 'team1',
              strengths: safeList(doi1, 'diem_manh', 'strengths'), weaknesses: safeList(doi1, 'diem_yeu', 'weaknesses') },
            { evalObj: doi2, name: t2Name, grade: t2Grade, score: t2Score, side: 'team2',
              strengths: safeList(doi2, 'diem_manh', 'strengths'), weaknesses: safeList(doi2, 'diem_yeu', 'weaknesses') },
          ].map((team, i) => (
            <Card key={i} animate delay={i * 0.08}>
              <div className="flex items-start justify-between gap-3 mb-4">
                <div>
                  <h3 className="text-xl font-bold text-text-primary">{team.name}</h3>
                  <Badge variant={team.side === 'team1' ? 'team1' : 'team2'} size="md" className="mt-1">
                    {safeText(team.evalObj, 'phong_cach', 'tactical_profile') || 'Chiến thuật'}
                  </Badge>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <GradeChip grade={team.grade} size="md" />
                  <span className="font-mono text-lg font-bold text-text-primary">{team.score}</span>
                </div>
              </div>
              {safeText(team.evalObj, 'nhan_xet_chien_thuat', 'tactical_comment', 'comment') && (
                <p className="text-sm text-text-secondary leading-relaxed mb-4 pb-4 border-b border-border">
                  {safeText(team.evalObj, 'nhan_xet_chien_thuat', 'tactical_comment', 'comment')}
                </p>
              )}
              {(team.strengths.length > 0 || team.weaknesses.length > 0) && (
                <div className="grid grid-cols-2 gap-4 mb-4 pb-4 border-b border-border">
                  {team.strengths.length > 0 && (
                    <div>
                      <p className="text-xs font-semibold text-emerald-400 uppercase tracking-wide mb-2">Điểm mạnh</p>
                      <ul className="space-y-1.5">{team.strengths.map((s, j) => <Bullet key={j} text={s} type="strength" />)}</ul>
                    </div>
                  )}
                  {team.weaknesses.length > 0 && (
                    <div>
                      <p className="text-xs font-semibold text-red-400 uppercase tracking-wide mb-2">Điểm yếu</p>
                      <ul className="space-y-1.5">{team.weaknesses.map((w, j) => <Bullet key={j} text={w} type="weakness" />)}</ul>
                    </div>
                  )}
                </div>
              )}
              <div className="space-y-2">
                {CATEGORY_META.map(cat => <CategoryBlock key={cat.key} cat={cat} evalObj={team.evalObj} />)}
              </div>
            </Card>
          ))}
        </div>

        {/* Section 3: Notable players */}
        {Object.keys(notable).length > 0 && (
          <Card animate delay={0.1}>
            <div className="flex items-center gap-3 mb-5">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center text-sm" style={{ background: 'rgba(245,158,11,0.15)' }}>⭐</div>
              <h2 className="text-xl font-bold text-text-primary">Cầu thủ nổi bật</h2>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="space-y-3">
                <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide">Đội 1</p>
                <PlayerCard player={notable.team1_best || notable.best_team1} type="best" side="team1" />
                <PlayerCard player={notable.team1_improve || notable.improve_team1} type="improve" side="team1" />
              </div>
              <div className="space-y-3">
                <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide">Đội 2</p>
                <PlayerCard player={notable.team2_best || notable.best_team2} type="best" side="team2" />
                <PlayerCard player={notable.team2_improve || notable.improve_team2} type="improve" side="team2" />
              </div>
            </div>
          </Card>
        )}

        {/* Section 4: Head-to-head */}
        {Object.keys(so_sanh).length > 0 && (
          <Card animate delay={0.12}>
            <div className="flex items-center gap-3 mb-5">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center text-sm" style={{ background: 'rgba(239,68,68,0.15)' }}>⚔️</div>
              <h2 className="text-xl font-bold text-text-primary">So sánh đối đầu</h2>
            </div>
            <div className="space-y-3">
              {HEAD2HEAD_CATS.map(cat => {
                const text = safeText(so_sanh, cat.key, `${cat.key}_comparison`)
                if (!text) return null
                return (
                  <div key={cat.key} className="flex gap-4 p-4 rounded-xl border border-border" style={{ background: 'var(--color-surface-2)' }}>
                    <span className="text-lg flex-shrink-0">{cat.icon}</span>
                    <div>
                      <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1">{cat.label}</p>
                      <p className="text-sm text-text-secondary leading-relaxed">{text}</p>
                    </div>
                  </div>
                )
              })}
            </div>
          </Card>
        )}

        {/* Section 5: Kết luận */}
        {ket_luan && (
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
            className="relative rounded-2xl p-6 overflow-hidden"
            style={{ background: 'linear-gradient(135deg, rgba(59,130,246,0.08), rgba(16,185,129,0.08))', border: '1px solid rgba(59,130,246,0.3)' }}>
            <div className="absolute top-0 right-0 w-48 h-48 rounded-full opacity-5"
              style={{ background: 'var(--color-team-1)', transform: 'translate(30%, -30%)' }} />
            <div className="relative">
              <h3 className="text-lg font-bold text-text-primary mb-3 flex items-center gap-2">🎯 Kết luận</h3>
              <p className="text-text-secondary leading-relaxed">{ket_luan}</p>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

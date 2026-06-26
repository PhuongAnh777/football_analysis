"""
ReportBuilder
=============
Assembles a structured ``match_report`` from a ``ThresholdEngine`` scored
report and a ``TacticalAnalyzer`` tactical report.

Output structure (6 sections + meta):
  meta                — formation, possession, dominant_team, styles
  mo_hinh_tran        — shape model: formation, block, width, possession
  press_and_recovery  — pressing H1/H2, recoveries by zone, PPDA
  cau_truc_doi_hinh   — compact trend, width delta with/without ball
  buildup_and_risk    — progressive pass %, turnovers final third
  cau_thu_then_chot   — top pressers, top width users, poor positioning
  insights            — 3–5 auto-generated actionable sentences per team
  match_narrative_data — flat key-value data for backward compat (LLM/adapter)
"""

from __future__ import annotations

from typing import Any

_LINE_MAP: dict[str, str] = {
    "DEF": "DEF", "MID": "MID", "FWD": "FWD",
    "DM":  "MID", "AM":  "MID", "SS":  "MID",
}


def _r2(v: Any) -> float:
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return 0.0


class ReportBuilder:
    """Build a structured match report from scored and tactical reports.

    Parameters
    ----------
    window_frames : int
        Frames per analysis window (default 720 = 30 s × 24 fps).
    """

    def __init__(self, window_frames: int = 720) -> None:
        self.window_frames = window_frames

    # ── public API ──────────────────────────────────────────────────────────

    def build(
        self,
        scored_report: dict,
        tactical_report: dict,
        total_frames: int | None = None,
    ) -> dict[str, Any]:
        """Assemble the full ``match_report``.

        Returns
        -------
        JSON-serialisable dict with keys:
        ``"meta"``, ``"mo_hinh_tran"``, ``"press_and_recovery"``,
        ``"cau_truc_doi_hinh"``, ``"buildup_and_risk"``,
        ``"cau_thu_then_chot"``, ``"insights"``,
        ``"match_narrative_data"``.
        """
        if total_frames is None:
            total_frames = self._estimate_total_frames(tactical_report)

        return {
            "meta":                 self._build_meta(scored_report, tactical_report, total_frames),
            "mo_hinh_tran":         self._build_mo_hinh_tran(tactical_report),
            "press_and_recovery":   self._build_press_and_recovery(tactical_report, total_frames),
            "cau_truc_doi_hinh":    self._build_cau_truc_doi_hinh(tactical_report, scored_report),
            "buildup_and_risk":     self._build_buildup_and_risk(tactical_report),
            "cau_thu_then_chot":    self._build_cau_thu_then_chot(scored_report, tactical_report),
            "insights":             self._build_insights(tactical_report),
            "match_narrative_data": self._build_narrative(tactical_report),
        }

    # ── tactical-report accessors (new + legacy key support) ────────────────

    def _compact(self, rpt: dict) -> dict:
        return rpt.get("compact") or rpt.get("compact_score") or {}

    def _pressing(self, rpt: dict) -> dict:
        return rpt.get("pressing") or rpt.get("pressing_intensity") or {}

    def _formation(self, rpt: dict) -> dict:
        return rpt.get("formation") or rpt.get("formation_adherence") or {}

    def _possession(self, rpt: dict) -> dict:
        return rpt.get("possession") or rpt.get("possession_stats") or {}

    def _def_line(self, rpt: dict) -> dict:
        return rpt.get("def_line") or {}

    def _team_width(self, rpt: dict) -> dict:
        return rpt.get("team_width") or {}

    def _compact_windows(self, rpt: dict, team_idx: int) -> list[dict]:
        c = self._compact(rpt)
        return c.get(f"team_{team_idx}") or c.get(str(team_idx)) or []

    def _formation_team(self, rpt: dict, team_idx: int) -> dict:
        f = self._formation(rpt)
        return f.get(f"team_{team_idx}") or f.get(str(team_idx)) or {}

    # ── total_frames estimation ──────────────────────────────────────────────

    def _estimate_total_frames(self, tactical_report: dict) -> int:
        max_start = 0
        for team_idx in (1, 2):
            for w in self._compact_windows(tactical_report, team_idx):
                max_start = max(max_start, w.get("window_start_frame", 0))
        for w in self._pressing(tactical_report).get("windows", []):
            max_start = max(max_start, w.get("window_start_frame", 0))
        return max_start + self.window_frames

    # ── meta ────────────────────────────────────────────────────────────────

    def _build_meta(
        self, scored: dict, tactical: dict, total_frames: int
    ) -> dict[str, Any]:
        summary = scored.get("match_summary", {})
        poss    = self._possession(tactical).get("possession", {})
        return {
            "formation_team_1":   summary.get(
                "formation_team_1",
                self._formation_team(tactical, 1).get("detected_formation", "unknown"),
            ),
            "formation_team_2":   summary.get(
                "formation_team_2",
                self._formation_team(tactical, 2).get("detected_formation", "unknown"),
            ),
            "possession_team_1":  _r2(poss.get("team_1", 0.0)),
            "possession_team_2":  _r2(poss.get("team_2", 0.0)),
            "dominant_team":      summary.get("dominant_team"),
            "def_style_team_1":   summary.get("def_style_team_1",   "mid_block"),
            "def_style_team_2":   summary.get("def_style_team_2",   "mid_block"),
            "width_style_team_1": summary.get("width_style_team_1", "medium"),
            "width_style_team_2": summary.get("width_style_team_2", "medium"),
            "total_frames":       int(total_frames),
        }

    # ── 1. MÔ HÌNH TRẬN ─────────────────────────────────────────────────────

    def _build_mo_hinh_tran(self, tactical: dict) -> dict[str, Any]:
        result: dict[str, Any] = {}
        poss = self._possession(tactical).get("possession", {})

        for team_idx in (1, 2):
            tk         = f"team_{team_idx}"
            form       = self._formation_team(tactical, team_idx)
            dl         = self._def_line(tactical).get(tk, {})
            ww         = self._team_width(tactical).get(tk, {})
            result[tk] = {
                "formation":      form.get("detected_formation", "unknown"),
                "formation_conf": _r2(form.get("confidence", 0.0)),
                "lines":          form.get("lines", {}),
                "possession_pct": _r2(poss.get(tk, 0.0)),
                "block_type":     dl.get("dominant_block", "mid_block"),
                "block_height_m": _r2(dl.get("overall_avg_m", 0.0)),
                "block_height_pct": _r2(dl.get("overall_avg_pct", 0.0)),
                "width_style":    ww.get("dominant_style", "medium"),
                "width_avg_m":    _r2(ww.get("overall_avg_m", 0.0)),
            }
        return result

    # ── 2. PRESS & RECOVERY ──────────────────────────────────────────────────

    def _build_press_and_recovery(
        self, tactical: dict, total_frames: int
    ) -> dict[str, Any]:
        press    = self._pressing(tactical)
        half_sum = press.get("half_summary", {})
        h1_int   = _r2(half_sum.get("half_1", {}).get("mean_intensity", 0.0))
        h2_int   = _r2(half_sum.get("half_2", {}).get("mean_intensity", 0.0))
        drop_pct = _r2((h2_int - h1_int) / max(h1_int, 1e-6) * 100) if h1_int else 0.0

        ppda_data = press.get("ppda") or {}

        result: dict[str, Any] = {}
        for team_idx in (1, 2):
            tk  = f"team_{team_idx}"
            rec = tactical.get("ball_recoveries", {}).get(tk, {})
            total_rec = int(rec.get("total_recoveries", 0))
            zones     = rec.get("recoveries_by_zone", {})
            opp_rec   = int(zones.get("opp_half", 0)) + int(zones.get("final_third", 0))
            opp_pct   = _r2(opp_rec / total_rec * 100) if total_rec else 0.0

            ppda_team = ppda_data.get(tk, {}).get("overall", {}) if ppda_data else {}
            ppda_val  = ppda_team.get("ppda") if ppda_team else None
            result[tk] = {
                "pressing_h1":           h1_int,
                "pressing_h2":           h2_int,
                "pressing_drop_pct":     drop_pct,
                "recoveries_total":      total_rec,
                "recoveries_opp_pct":    opp_pct,
                "ppda":                  _r2(ppda_val) if ppda_val is not None else None,
                "ppda_label":            ppda_team.get("intensity_label") if ppda_team else None,
                "ppda_half1":            ppda_data.get(tk, {}).get("half_1_avg_ppda") if ppda_data else None,
                "ppda_half2":            ppda_data.get(tk, {}).get("half_2_avg_ppda") if ppda_data else None,
            }
        return result

    # ── 3. CẤU TRÚC ĐỘI HÌNH ────────────────────────────────────────────────

    def _build_cau_truc_doi_hinh(
        self, tactical: dict, scored: dict
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        phase_all = self._compact(tactical).get("phase_summary", {})
        for team_idx in (1, 2):
            tk      = f"team_{team_idx}"
            c_wins  = self._compact_windows(tactical, team_idx)
            c_avg   = (
                sum(w.get("mean_area", 0) for w in c_wins) / len(c_wins)
                if c_wins else 0.0
            )
            phase   = phase_all.get(tk, {})
            ww      = self._team_width(tactical).get(tk, {})
            w_ball  = _r2(ww.get("width_with_ball",    0.0))
            w_no    = _r2(ww.get("width_without_ball", 0.0))

            result[tk] = {
                "compact_trend":          self._compact_trend(tactical, team_idx),
                "compact_avg_m2":         _r2(c_avg),
                "compact_attacking_m2":   phase.get("attacking_avg_m2"),
                "compact_defending_m2":   phase.get("defending_avg_m2"),
                "compact_phase_comment":  self._compact_phase_comment(phase),
                "width_with_ball_m":      w_ball,
                "width_without_ball_m":   w_no,
                "width_delta_m":          _r2(w_ball - w_no),
            }
        return result

    # ── 4. BUILD-UP & RỦI RO ────────────────────────────────────────────────

    def _build_buildup_and_risk(self, tactical: dict) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for team_idx in (1, 2):
            tk       = f"team_{team_idx}"
            to_data  = tactical.get("turnovers", {}).get(tk, {})
            p_data   = (tactical.get("passing") or {}).get(tk)
            pass_info: dict[str, Any] = {}
            if isinstance(p_data, dict):
                pass_info = {
                    "total_passes":          int(p_data.get("total_passes", 0)),
                    "progressive_pass_pct":  _r2(p_data.get("progressive_pass_pct", 0.0)),
                    "network_density":       _r2(p_data.get("network_density", 0.0)),
                    "top_passer_id":         p_data.get("top_passer_id"),
                    "top_receiver_id":       p_data.get("top_receiver_id"),
                }
            result[tk] = {
                **pass_info,
                "turnovers_final_third":    int(to_data.get("total_turnovers_in_final_third", 0)),
                "high_risk_count":          int(to_data.get("high_risk_count",           0)),
                "low_risk_count":           int(to_data.get("low_risk_count",            0)),
                "high_risk_rate_pct":       _r2(to_data.get("high_risk_rate_pct",        0.0)),
                "avg_distance_to_goal_m":   _r2(to_data.get("avg_distance_to_goal_m",    0.0)),
                "avg_transition_potential": _r2(to_data.get("avg_transition_potential",  0.0)),
            }
        return result

    # ── 5. CẦU THỦ THEN CHỐT ────────────────────────────────────────────────

    def _build_cau_thu_then_chot(
        self, scored: dict, tactical: dict
    ) -> dict[str, Any]:
        """Top pressers, top width users, poor positioning (DEF) per team."""
        result: dict[str, Any] = {}

        role_lookup: dict[int, str] = {}
        for team_idx in (1, 2):
            for raw_label, ids in self._formation_team(tactical, team_idx).get("lines", {}).items():
                mapped = _LINE_MAP.get(raw_label, "MID")
                for pid in ids:
                    role_lookup[int(pid)] = mapped

        for team_str, players in scored.get("player_scores", {}).items():
            if not players:
                result[f"team_{team_str}"] = {
                    "top_pressers":    [],
                    "top_width_users": [],
                    "poor_positioning": [],
                }
                continue

            def _entry(tid_str: str, pscores: dict, key: str) -> dict:
                return {
                    "track_id": int(tid_str),
                    "score":    _r2(pscores.get(key, 0.0)),
                    "role":     role_lookup.get(int(tid_str), "MID"),
                }

            sorted_pressing = sorted(
                players.items(),
                key=lambda kv: kv[1].get("pressing_score", 0.0),
                reverse=True,
            )
            sorted_width = sorted(
                [
                    (tid, p) for tid, p in players.items()
                    if role_lookup.get(int(tid), "MID") in ("MID", "FWD")
                ],
                key=lambda kv: kv[1].get("width_contrib_score", 0.0),
                reverse=True,
            )
            def_players = [
                (tid, p) for tid, p in players.items()
                if role_lookup.get(int(tid), "MID") == "DEF"
            ]
            sorted_def_poor = sorted(
                def_players,
                key=lambda kv: kv[1].get("def_positioning_score", 0.0),
            )

            result[f"team_{team_str}"] = {
                "top_pressers": [
                    _entry(tid, p, "pressing_score")
                    for tid, p in sorted_pressing[:3]
                ],
                "top_width_users": [
                    _entry(tid, p, "width_contrib_score")
                    for tid, p in sorted_width[:3]
                ],
                "poor_positioning": [
                    _entry(tid, p, "def_positioning_score")
                    for tid, p in sorted_def_poor[:3]
                ],
            }
        return result

    # ── 6. INSIGHTS (auto-generated actionable sentences) ────────────────────

    def _build_insights(self, tactical: dict) -> dict[str, list[str]]:
        return {
            "team_1": self._gen_team_insights(tactical, 1),
            "team_2": self._gen_team_insights(tactical, 2),
        }

    def _gen_team_insights(self, tactical: dict, team_idx: int) -> list[str]:
        tk       = f"team_{team_idx}"
        insights: list[str] = []

        # 1. Pressing / PPDA trend across video halves
        pr = self._build_press_and_recovery(tactical, 0).get(tk, {})
        ppda_h1 = pr.get("ppda_half1")
        ppda_h2 = pr.get("ppda_half2")
        if ppda_h1 is not None and ppda_h2 is not None and ppda_h1 > 0:
            drop = (ppda_h2 - ppda_h1) / ppda_h1 * 100
            if abs(drop) >= 10:
                verb = "giảm" if drop < 0 else "tăng"
                insights.append(
                    f"PPDA {verb} {abs(drop):.0f}% "
                    f"từ nửa đầu video ({ppda_h1:.1f}) → nửa sau ({ppda_h2:.1f})"
                )
        else:
            half_sum = self._pressing(tactical).get("half_summary", {})
            h1 = float(half_sum.get("half_1", {}).get("mean_intensity", 0.0))
            h2 = float(half_sum.get("half_2", {}).get("mean_intensity", 0.0))
            if h1 > 0:
                drop = (h2 - h1) / h1 * 100
                if abs(drop) >= 10:
                    verb = "giảm" if drop < 0 else "tăng"
                    insights.append(
                        f"Pressing proximity {verb} {abs(drop):.0f}% "
                        f"từ nửa đầu video ({h1:.2f}) → nửa sau ({h2:.2f})"
                    )

        # 2. Recovery zone
        rec   = tactical.get("ball_recoveries", {}).get(tk, {})
        total = int(rec.get("total_recoveries", 0))
        if total > 0:
            z   = rec.get("recoveries_by_zone", {})
            opp = int(z.get("opp_half", 0)) + int(z.get("final_third", 0))
            pct = opp / total * 100
            label = "counter-press hiệu quả" if pct > 50 else "cướp bóng chủ yếu ở sân nhà"
            insights.append(
                f"{pct:.0f}% ({opp}/{total}) lần thu hồi bóng ở phần sân đối phương"
                f" → {label}"
            )

        # 3. Turnovers final third — contextual risk classification
        to_data  = tactical.get("turnovers", {}).get(tk, {})
        total_to = int(to_data.get("total_turnovers_in_final_third", 0))
        high_risk  = int(to_data.get("high_risk_count",      0))
        low_risk   = int(to_data.get("low_risk_count",       0))
        hr_rate    = float(to_data.get("high_risk_rate_pct", 0.0))
        avg_dist   = float(to_data.get("avg_distance_to_goal_m",   0.0))
        if total_to > 0:
            insights.append(
                f"{total_to} lần mất bóng ở 1/3 cuối: "
                f"{high_risk} High-Risk / {low_risk} Low-Risk "
                f"({hr_rate:.0f}% chuyển đổi nguy hiểm, "
                f"cách khung thành TB {avg_dist:.1f} m)"
            )
        else:
            insights.append(
                "Không mất bóng ở 1/3 cuối sân — xây dựng bóng thận trọng"
            )

        # 4. Width delta with/without ball
        ww    = self._team_width(tactical).get(tk, {})
        w_b   = float(ww.get("width_with_ball",    0.0))
        w_nb  = float(ww.get("width_without_ball", 0.0))
        delta = w_b - w_nb
        if abs(delta) >= 3:
            verb = "mở rộng" if delta > 0 else "thu hẹp"
            insights.append(
                f"Đội hình {verb} {abs(delta):.1f} m khi có bóng"
                f" ({w_b:.1f} m) so với không bóng ({w_nb:.1f} m)"
            )

        # 5. Progressive passing
        p = (tactical.get("passing") or {}).get(tk)
        if isinstance(p, dict):
            prog = float(p.get("progressive_pass_pct", 0.0))
            if prog > 35:
                insights.append(
                    f"{prog:.1f}% đường chuyền tiến bộ"
                    f" → lối chơi tấn công trực tiếp"
                )
            else:
                insights.append(
                    f"{prog:.1f}% đường chuyền tiến bộ"
                    f" — ưu tiên giữ bóng an toàn"
                )

        # 6. Compact trend + phase split
        trend  = self._compact_trend(tactical, team_idx)
        c_wins = self._compact_windows(tactical, team_idx)
        phase  = self._compact(tactical).get("phase_summary", {}).get(tk, {})
        phase_comment = self._compact_phase_comment(phase)
        if phase_comment:
            insights.append(phase_comment)
        elif c_wins and trend != "stable":
            avg_m2 = sum(w.get("mean_area", 0) for w in c_wins) / len(c_wins)
            label  = "ngày càng chặt chẽ" if trend == "improving" else "bị kéo giãn dần"
            insights.append(
                f"Độ compact {label} qua video"
                f" (hull area trung bình {avg_m2:.1f} m²)"
            )

        return insights[:5]

    # ── narrative data (flat key-value for LLM / result_adapter) ────────────

    def _build_narrative(self, tactical: dict) -> dict[str, Any]:
        press    = self._pressing(tactical)
        half_sum = press.get("half_summary", {})
        poss     = self._possession(tactical)
        spd      = poss.get("avg_speed", {})
        zones    = poss.get("speed_zones", {})

        def _rec(tk: str) -> dict:
            return tactical.get("ball_recoveries", {}).get(tk, {})

        def _to(tk: str) -> dict:
            return tactical.get("turnovers", {}).get(tk, {})

        def _pass(tk: str):
            p = tactical.get("passing")
            return p.get(tk) if p else None

        def _opp_half_pct(tk: str) -> float:
            d     = _rec(tk)
            total = int(d.get("total_recoveries", 0))
            z     = d.get("recoveries_by_zone", {})
            opp   = int(z.get("opp_half", 0)) + int(z.get("final_third", 0))
            return _r2(opp / total * 100) if total else 0.0

        p1 = _pass("team_1")
        p2 = _pass("team_2")

        return {
            "compact_attacking_m2_team_1":     self._compact(tactical).get("phase_summary", {}).get("team_1", {}).get("attacking_avg_m2"),
            "compact_defending_m2_team_1":     self._compact(tactical).get("phase_summary", {}).get("team_1", {}).get("defending_avg_m2"),
            "compact_attacking_m2_team_2":     self._compact(tactical).get("phase_summary", {}).get("team_2", {}).get("attacking_avg_m2"),
            "compact_defending_m2_team_2":     self._compact(tactical).get("phase_summary", {}).get("team_2", {}).get("defending_avg_m2"),
            "ppda_team_1":                     _r2(self._pressing(tactical).get("ppda", {}).get("team_1", {}).get("overall", {}).get("ppda"))
            if self._pressing(tactical).get("ppda") else None,
            "ppda_team_2":                     _r2(self._pressing(tactical).get("ppda", {}).get("team_2", {}).get("overall", {}).get("ppda"))
            if self._pressing(tactical).get("ppda") else None,
            "pressing_intensity_half1":        _r2(half_sum.get("half_1", {}).get("mean_intensity", 0.0)),
            "pressing_intensity_half2":        _r2(half_sum.get("half_2", {}).get("mean_intensity", 0.0)),
            "compact_trend_team_1":            self._compact_trend(tactical, 1),
            "compact_trend_team_2":            self._compact_trend(tactical, 2),
            "speed_team_1":                    _r2(spd.get("team_1", {}).get("overall", 0.0)),
            "speed_team_2":                    _r2(spd.get("team_2", {}).get("overall", 0.0)),
            "sprint_pct_team_1":               _r2(zones.get("team_1", {}).get("sprinting", 0.0)),
            "sprint_pct_team_2":               _r2(zones.get("team_2", {}).get("sprinting", 0.0)),
            "formation_team_1":                self._formation_team(tactical, 1).get("detected_formation", "unknown"),
            "formation_team_2":                self._formation_team(tactical, 2).get("detected_formation", "unknown"),
            "def_line_avg_team_1":             _r2(self._def_line(tactical).get("team_1", {}).get("overall_avg_m", 0.0)),
            "def_line_avg_team_2":             _r2(self._def_line(tactical).get("team_2", {}).get("overall_avg_m", 0.0)),
            "def_line_trend_team_1":           self._def_line(tactical).get("team_1", {}).get("trend", "stable"),
            "def_line_trend_team_2":           self._def_line(tactical).get("team_2", {}).get("trend", "stable"),
            "width_avg_team_1":                _r2(self._team_width(tactical).get("team_1", {}).get("overall_avg_m", 0.0)),
            "width_avg_team_2":                _r2(self._team_width(tactical).get("team_2", {}).get("overall_avg_m", 0.0)),
            "width_with_ball_team_1":          _r2(self._team_width(tactical).get("team_1", {}).get("width_with_ball", 0.0)),
            "width_with_ball_team_2":          _r2(self._team_width(tactical).get("team_2", {}).get("width_with_ball", 0.0)),
            "width_without_ball_team_1":       _r2(self._team_width(tactical).get("team_1", {}).get("width_without_ball", 0.0)),
            "width_without_ball_team_2":       _r2(self._team_width(tactical).get("team_2", {}).get("width_without_ball", 0.0)),
            "recoveries_team_1":               int(_rec("team_1").get("total_recoveries", 0)),
            "recoveries_team_2":               int(_rec("team_2").get("total_recoveries", 0)),
            "opp_half_recovery_pct_team_1":    _opp_half_pct("team_1"),
            "opp_half_recovery_pct_team_2":    _opp_half_pct("team_2"),
            "turnovers_final_third_team_1":    int(_to("team_1").get("total_turnovers_in_final_third", 0)),
            "turnovers_final_third_team_2":    int(_to("team_2").get("total_turnovers_in_final_third", 0)),
            "high_risk_count_team_1":          int(_to("team_1").get("high_risk_count",      0)),
            "high_risk_count_team_2":          int(_to("team_2").get("high_risk_count",      0)),
            "high_risk_rate_pct_team_1":       _r2(_to("team_1").get("high_risk_rate_pct",   0.0)),
            "high_risk_rate_pct_team_2":       _r2(_to("team_2").get("high_risk_rate_pct",   0.0)),
            "avg_distance_to_goal_team_1":     _r2(_to("team_1").get("avg_distance_to_goal_m", 0.0)),
            "avg_distance_to_goal_team_2":     _r2(_to("team_2").get("avg_distance_to_goal_m", 0.0)),
            "avg_transition_potential_team_1": _r2(_to("team_1").get("avg_transition_potential", 0.0)),
            "avg_transition_potential_team_2": _r2(_to("team_2").get("avg_transition_potential", 0.0)),
            "total_passes_team_1":             int(p1.get("total_passes", 0)) if isinstance(p1, dict) else None,
            "total_passes_team_2":             int(p2.get("total_passes", 0)) if isinstance(p2, dict) else None,
            "progressive_pct_team_1":          _r2(p1.get("progressive_pass_pct", 0.0)) if isinstance(p1, dict) else None,
            "progressive_pct_team_2":          _r2(p2.get("progressive_pass_pct", 0.0)) if isinstance(p2, dict) else None,
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compact_phase_comment(phase: dict) -> str | None:
        att = phase.get("attacking_avg_m2")
        def_ = phase.get("defending_avg_m2")
        if att is None or def_ is None:
            return None
        delta = float(att) - float(def_)
        if abs(delta) < 20:
            return (
                f"Hull area tấn công {att:.0f} m² vs phòng ngự {def_:.0f} m² "
                f"— đội hình ổn định giữa hai phase"
            )
        if delta > 0:
            return (
                f"Khi tấn công hull area {att:.0f} m² (rộng hơn {delta:.0f} m² "
                f"so với phase phòng ngự {def_:.0f} m²)"
            )
        return (
            f"Khi phòng ngự hull area {def_:.0f} m² (chặt hơn {abs(delta):.0f} m² "
            f"so với phase tấn công {att:.0f} m²)"
        )

    def _compact_trend(self, tactical: dict, team_idx: int) -> str:
        windows = self._compact_windows(tactical, team_idx)
        n = len(windows)
        if n < 2:
            return "stable"
        half  = max(n // 2, 1)
        avg1  = sum(w.get("mean_area", 0.0) for w in windows[:half]) / half
        avg2  = sum(w.get("mean_area", 0.0) for w in windows[half:]) / max(n - half, 1)
        diff  = avg2 - avg1
        if diff < -1.0:
            return "improving"
        if diff >  1.0:
            return "declining"
        return "stable"

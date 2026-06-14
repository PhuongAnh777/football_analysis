"""
Normalise pipeline outputs into the shape expected by the React frontend.

The frontend expects top-level keys ``teams``, ``players``, ``evaluation``,
``charts``, ``timeline``, and ``notable_players`` in addition to the raw
``match_report``.
"""

from __future__ import annotations

from typing import Any


# ── helpers ──────────────────────────────────────────────────────────────────

def _r2(v: Any) -> float:
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return 0.0


def _team_metrics(
    team_idx: int,
    tactical_report: dict,
    match_report: dict,
    scored_report: dict,
) -> dict[str, Any]:
    tk        = f"team_{team_idx}"
    meta      = match_report.get("meta", {})
    narrative = match_report.get("match_narrative_data", {})
    press_rec = match_report.get("press_and_recovery", {}).get(tk, {})
    cau_truc  = match_report.get("cau_truc_doi_hinh",  {}).get(tk, {})
    mo_hinh   = match_report.get("mo_hinh_tran",       {}).get(tk, {})

    compact_windows = (
        tactical_report.get("compact", {}).get(tk)
        or tactical_report.get("compact_score", {}).get(tk)
        or []
    )
    avg_compact = (
        sum(w.get("mean_area", w.get("compact_score", 0)) for w in compact_windows) / len(compact_windows)
        if compact_windows else 0.0
    )

    to_data   = tactical_report.get("turnovers", {}).get(tk, {})
    pass_data = (tactical_report.get("passing") or {}).get(tk, {})

    return {
        "possession":           float(meta.get(f"possession_team_{team_idx}", 0)),
        "compact_score":        round(avg_compact, 2),
        "compact_scored":       float(cau_truc.get("compact_score", 0)),
        "compact_trend":        cau_truc.get("compact_trend", "stable"),
        "pressing_h1":          float(press_rec.get("pressing_h1", 0)),
        "pressing_h2":          float(press_rec.get("pressing_h2", 0)),
        "pressing_drop_pct":    float(press_rec.get("pressing_drop_pct", 0)),
        "avg_speed":            float(narrative.get(f"speed_team_{team_idx}", 0)),
        "sprint_pct":           float(narrative.get(f"sprint_pct_team_{team_idx}", 0)),
        "defensive_line_height": float(narrative.get(f"def_line_avg_team_{team_idx}", 0)),
        "block_type":           mo_hinh.get("block_type", "mid_block"),
        "width":                float(narrative.get(f"width_avg_team_{team_idx}", 0)),
        "width_with_ball":      float(narrative.get(f"width_with_ball_team_{team_idx}", 0)),
        "width_without_ball":   float(narrative.get(f"width_without_ball_team_{team_idx}", 0)),
        "width_delta":          float(cau_truc.get("width_delta_m", 0)),
        "ball_recoveries":      int(press_rec.get("recoveries_total", 0)),
        "recoveries_opp_pct":   float(press_rec.get("recoveries_opp_pct", 0)),
        "turnovers_final_third":    int(to_data.get("total_turnovers_in_final_third", 0)),
        "high_risk_count":          int(to_data.get("high_risk_count",      0)),
        "high_risk_rate_pct":       float(to_data.get("high_risk_rate_pct", 0)),
        "avg_distance_to_goal_m":   float(to_data.get("avg_distance_to_goal_m", 0)),
        "avg_transition_potential": float(to_data.get("avg_transition_potential", 0)),
        "forward_passes_pct":   float(
            pass_data.get("progressive_pass_pct", 0) if isinstance(pass_data, dict) else 0
        ),
    }


def _build_teams(
    tactical_report: dict,
    match_report: dict,
    scored_report: dict,
    llm_eval: dict | None,
) -> list[dict]:
    teams: list[dict] = []
    meta      = match_report.get("meta", {})
    danh_gia  = (llm_eval or {}).get("danh_gia_doi", {})
    cau_truc  = match_report.get("cau_truc_doi_hinh", {})
    insights  = match_report.get("insights", {})

    for team_idx in (1, 2):
        tk       = f"team_{team_idx}"
        llm_team = danh_gia.get(f"doi_{team_idx}", {})
        c_score  = float(cau_truc.get(tk, {}).get("compact_score", 50))

        teams.append({
            "name":             llm_team.get("ten") or f"Đội {team_idx}",
            "formation":        meta.get(f"formation_team_{team_idx}", "unknown"),
            "tactical_profile": llm_team.get("chien_thuat", ""),
            "insights":         insights.get(tk, []),
            "metrics":          _team_metrics(team_idx, tactical_report, match_report, scored_report),
        })
    return teams


def _build_players(
    match_report: dict,
    scored_report: dict,
) -> list[dict]:
    players: list[dict] = []
    cau_thu = match_report.get("cau_thu_then_chot", {})
    raw_scores = scored_report.get("player_scores", {})

    # Collect all known player track_ids from scored_report for a full roster
    for team_str, team_raw in raw_scores.items():
        team_id    = int(team_str)
        team_chot  = cau_thu.get(f"team_{team_str}", {})
        top_press  = {p["track_id"] for p in team_chot.get("top_pressers",    [])}
        top_width  = {p["track_id"] for p in team_chot.get("top_width_users", [])}
        poor_pos   = {p["track_id"] for p in team_chot.get("poor_positioning", [])}

        roster_items = sorted(team_raw.items(), key=lambda item: int(item[0]))
        for squad_num, (tid_str, raw) in enumerate(roster_items, start=1):
            tid = int(tid_str)
            tags: list[str] = []
            if tid in top_press:
                tags.append("top_presser")
            if tid in top_width:
                tags.append("top_width")
            if tid in poor_pos:
                tags.append("poor_positioning")

            players.append({
                "team":         team_id,
                "team_id":      team_id,
                "track_id":     tid,
                "squad_number": squad_num,
                "display_name": f"Cầu thủ {squad_num}",
                "position":     raw.get("role", "MID"),
                "tags":         tags,
                "pressing":     raw.get("pressing_score",        0),
                "width_contrib":raw.get("width_contrib_score",   0),
                "def_position": raw.get("def_positioning_score", 0),
                "speed":        raw.get("speed_score",           0),
                "activity":     raw.get("activity_score",        0),
            })
    return players


def _player_label_map(players: list[dict]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for player in players:
        track_id = player.get("track_id")
        if track_id is None:
            continue
        name = player.get("display_name") or f"Cầu thủ {player.get('squad_number', '?')}"
        role = player.get("position")
        labels[int(track_id)] = f"{name} ({role})" if role else name
    return labels


def _build_timeline(
    passing_events: list[dict],
    players: list[dict] | None = None,
) -> list[dict]:
    labels   = _player_label_map(players or [])
    timeline: list[dict] = []
    for ev in passing_events:
        passer_id   = ev.get("passer_id")
        receiver_id = ev.get("receiver_id")
        passer   = labels.get(int(passer_id),   "Cầu thủ ?") if passer_id   is not None else "—"
        receiver = labels.get(int(receiver_id), "Cầu thủ ?") if receiver_id is not None else "—"
        timeline.append({
            "frame":       ev.get("frame", 0),
            "team":        max(0, int(ev.get("team", 1)) - 1),
            "type":        "pass",
            "description": f"Chuyền bóng {passer} → {receiver}",
        })
    return timeline


def _build_notable_players(
    llm_eval: dict | None, match_report: dict
) -> dict:
    # Prefer LLM player data
    if llm_eval and llm_eval.get("danh_gia_cau_thu"):
        dc = llm_eval["danh_gia_cau_thu"]
        notable: dict[str, Any] = {}
        for team_idx in (1, 2):
            doi_key = f"doi_{team_idx}"
            prefix  = f"team{team_idx}"
            press   = dc.get(doi_key, {}).get("pressing_tot")
            width   = dc.get(doi_key, {}).get("width_tot")
            poor    = dc.get(doi_key, {}).get("can_cai_thien")
            if press:
                notable[f"{prefix}_best_presser"] = _player_card(press)
            if width:
                notable[f"{prefix}_best_width"] = _player_card(width)
            if poor:
                notable[f"{prefix}_improve"] = _player_card(poor)
        return notable

    # Fallback: use cau_thu_then_chot from match_report
    notable = {}
    cau_thu = match_report.get("cau_thu_then_chot", {})
    for team_idx in (1, 2):
        tk     = f"team_{team_idx}"
        prefix = f"team{team_idx}"
        team   = cau_thu.get(tk, {})
        pressers = team.get("top_pressers", [])
        widths   = team.get("top_width_users", [])
        poor     = team.get("poor_positioning", [])
        if pressers:
            notable[f"{prefix}_best_presser"] = {
                "track_id": pressers[0]["track_id"],
                "reason":   f"Cầu thủ #{pressers[0]['track_id']} — pressing cao nhất đội {team_idx}",
            }
        if widths:
            notable[f"{prefix}_best_width"] = {
                "track_id": widths[0]["track_id"],
                "reason":   f"Cầu thủ #{widths[0]['track_id']} — khai thác biên hiệu quả nhất",
            }
        if poor:
            notable[f"{prefix}_improve"] = {
                "track_id":       poor[0]["track_id"],
                "reason":         f"Cầu thủ #{poor[0]['track_id']} — cần cải thiện kỷ luật vị trí hàng thủ",
                "recommendation": "Tập trung vào kỷ luật chiều sâu hàng thủ.",
            }
    return notable


def _player_card(entry: dict | None) -> dict | None:
    if not entry:
        return None
    return {
        "track_id":       entry.get("track_id"),
        "reason":         entry.get("ly_do") or entry.get("reason"),
        "recommendation": entry.get("khuyen_nghi") or entry.get("recommendation"),
        "highlights":     entry.get("chi_so_noi_bat"),
    }


def _build_fallback_evaluation(match_report: dict) -> dict:
    """Structured evaluation from match_report when LLM is unavailable."""
    meta     = match_report.get("meta", {})
    narrative = match_report.get("match_narrative_data", {})
    insights  = match_report.get("insights", {})
    mo_hinh   = match_report.get("mo_hinh_tran", {})
    press_rec = match_report.get("press_and_recovery", {})

    teams_out: dict[str, Any] = {}

    for team_idx in (1, 2):
        tk        = f"team_{team_idx}"
        formation = mo_hinh.get(tk, {}).get("formation", meta.get(f"formation_team_{team_idx}", "unknown"))
        possession = meta.get(f"possession_team_{team_idx}", 0)
        pr        = press_rec.get(tk, {})
        h1        = pr.get("pressing_h1", 0)
        h2        = pr.get("pressing_h2", 0)
        drop      = pr.get("pressing_drop_pct", 0)
        rec_total = pr.get("recoveries_total", 0)
        rec_opp   = pr.get("recoveries_opp_pct", 0)
        to_total  = narrative.get(f"turnovers_final_third_team_{team_idx}", 0)
        hr_rate   = narrative.get(f"high_risk_rate_pct_team_{team_idx}", 0)
        high_risk_ct = narrative.get(f"high_risk_count_team_{team_idx}", 0)
        prog_pct  = narrative.get(f"progressive_pct_team_{team_idx}")

        teams_out[f"doi_{team_idx}"] = {
            "ten":          f"Đội {team_idx}",
            "so_do":        formation,
            "chien_thuat":  (
                f"Sơ đồ {formation}, kiểm soát bóng {possession:.1f}%."
            ),
            "pressing":     (
                f"H1: {h1:.2f} | H2: {h2:.2f} "
                f"({'giảm' if drop < 0 else 'tăng'} {abs(drop):.0f}%)"
                if h1 else "Không có dữ liệu pressing."
            ),
            "doi_hinh":     (
                f"Compact trend: {narrative.get(f'compact_trend_team_{team_idx}', 'stable')}"
            ),
            "hang_thu":     (
                f"Hàng thủ trung bình {narrative.get(f'def_line_avg_team_{team_idx}', 0):.1f} m, "
                f"xu hướng {narrative.get(f'def_line_trend_team_{team_idx}', 'stable')}"
            ),
            "xay_dung":     (
                f"Progressive pass: {prog_pct:.1f}%." if prog_pct is not None
                else "Không đủ dữ liệu chuyền bóng."
            ),
            "diem_manh":    [i for i in insights.get(tk, []) if "tăng" in i or "%" in i or "counter" in i][:2]
                            or [f"Thu hồi bóng {rec_total} lần ({rec_opp:.0f}% ở sân đối phương)"],
            "diem_yeu":     [f"Mất bóng {to_total} lần ở 1/3 cuối ({high_risk_ct} High-Risk, {hr_rate:.0f}% chuyển đổi nguy hiểm)"],
            "khuyen_nghi_hlv": ["Cải thiện pressing ở hiệp 2", "Tăng cường kỷ luật vị trí hàng thủ"],
            "insights":     insights.get(tk, []),
        }

    dom = meta.get("dominant_team")
    return {
        "tong_quan_tran_dau": {
            "nhan_xet_chung": (
                f"Đội hình {meta.get('formation_team_1')} vs {meta.get('formation_team_2')}. "
                f"Kiểm soát bóng {meta.get('possession_team_1', 0):.1f}% — "
                f"{meta.get('possession_team_2', 0):.1f}%."
            ),
            "doi_noi_bat": dom,
            "ly_do": f"Đội {dom} có compact và pressing tốt hơn." if dom else None,
        },
        "danh_gia_doi":     teams_out,
        "doi_1":            teams_out["doi_1"],
        "doi_2":            teams_out["doi_2"],
        "so_sanh_doi_dau": {
            "pressing": (
                f"H1: {press_rec.get('team_1', {}).get('pressing_h1', 0):.2f} / "
                f"{press_rec.get('team_2', {}).get('pressing_h1', 0):.2f} | "
                f"H2: {press_rec.get('team_1', {}).get('pressing_h2', 0):.2f} / "
                f"{press_rec.get('team_2', {}).get('pressing_h2', 0):.2f}"
            ),
            "kiem_soat_bong": (
                f"Đội 1 {meta.get('possession_team_1', 0):.1f}% — "
                f"Đội 2 {meta.get('possession_team_2', 0):.1f}%"
            ),
        },
        "ket_luan": (
            f"Trận đấu nghiêng về Đội {dom} về mặt compact và pressing."
            if dom else "Hai đội cân bằng về chỉ số chiến thuật."
        ),
    }


def normalize_evaluation(
    llm_eval: dict | None,
    match_report: dict,
) -> dict:
    if llm_eval and llm_eval.get("danh_gia_doi"):
        out = dict(llm_eval)
        out["doi_1"] = llm_eval["danh_gia_doi"].get("doi_1", {})
        out["doi_2"] = llm_eval["danh_gia_doi"].get("doi_2", {})
        if llm_eval.get("tong_quan_tran_dau") and not out.get("overview"):
            out["overview"] = llm_eval["tong_quan_tran_dau"]
        return out
    if llm_eval and not llm_eval.get("warning"):
        return llm_eval
    return _build_fallback_evaluation(match_report)


def adapt_api_result(
    match_report: dict,
    tactical_report: dict,
    scored_report: dict,
    charts: dict,
    passing_events: list[dict],
    llm_eval: dict | None,
    fps: float,
) -> dict:
    evaluation = normalize_evaluation(llm_eval, match_report)
    players    = _build_players(match_report, scored_report)
    return {
        "evaluation":      evaluation,
        "match_report":    match_report,
        "charts":          charts,
        "teams":           _build_teams(tactical_report, match_report, scored_report, llm_eval),
        "players":         players,
        "timeline":        _build_timeline(passing_events, players),
        "notable_players": _build_notable_players(llm_eval, match_report),
        "fps":             fps,
    }


def build_streamlit_analysis_result(tactical_report: dict) -> dict[int, list[dict]]:
    """Build time-series samples for the Streamlit Plotly charts."""
    result: dict[int, list] = {1: [], 2: []}

    for team_idx in (1, 2):
        tk = f"team_{team_idx}"
        compact_w = (
            tactical_report.get("compact", {}).get(tk)
            or []
        )
        press_w   = tactical_report.get("pressing", {}).get("windows", [])
        poss      = tactical_report.get("possession", {})
        speeds    = poss.get("avg_speed", {}).get(tk, {})
        team_speed = float(speeds.get("overall", 0)) if isinstance(speeds, dict) else 0.0

        formation_info = (
            tactical_report.get("formation", {}).get(tk)
            or tactical_report.get("formation_adherence", {}).get(tk)
            or {}
        )
        default_formation = formation_info.get("detected_formation", "unknown")

        press_by_frame: dict[int, float] = {}
        for w in press_w:
            if w.get("pressing_team") == team_idx:
                press_by_frame[w["window_start_frame"]] = w.get("intensity", 0)

        for w in compact_w:
            frame = w.get("window_start_frame", 0)
            result[team_idx].append({
                "frame":     frame,
                "formation": w.get("formation") or default_formation,
                "compact":   w.get("mean_area", w.get("compact_score", 0)),
                "pressing":  press_by_frame.get(frame, 0),
                "avg_speed": team_speed,
            })

        if not result[team_idx]:
            result[team_idx].append({
                "frame":     0,
                "formation": default_formation,
                "compact":   0,
                "pressing":  0,
                "avg_speed": team_speed,
            })

    return result

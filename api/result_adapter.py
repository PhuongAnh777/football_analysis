"""
Normalise pipeline outputs into the shape expected by the React frontend.

The frontend expects top-level keys ``teams``, ``players``, ``evaluation``,
``charts``, ``timeline``, and ``notable_players`` in addition to the raw
``match_report``.
"""

from __future__ import annotations

from typing import Any


_PITCH_LENGTH_M = 105.0


# ── helpers ──────────────────────────────────────────────────────────────────

def _def_line_depth_pct(
    depth_m: float,
    pitch_length: float = _PITCH_LENGTH_M,
) -> float:
    """Depth from own goal line as % of pitch length (0–100)."""
    clamped = max(0.0, min(float(depth_m), pitch_length))
    return round(clamped / pitch_length * 100, 1)


def _r2(v: Any) -> float:
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return 0.0


def _compute_pressing_intensity(
    team_idx: int,
    press_rec: dict,
    scored_report: dict,
    metrics: dict | None = None,
) -> float:
    """Return a 0–100 pressing score for frontend charts."""
    tk = f"team_{team_idx}"
    team_scores = scored_report.get("team_scores", {}).get(tk, {})
    pressing_score = team_scores.get("pressing_score")
    if pressing_score is not None:
        return float(pressing_score)

    metrics = metrics or {}
    h1 = float(metrics.get("pressing_h1", press_rec.get("pressing_h1", 0)) or 0)
    h2 = float(metrics.get("pressing_h2", press_rec.get("pressing_h2", 0)) or 0)
    if h1 or h2:
        avg = (h1 + h2) / 2 if h1 and h2 else h1 or h2
        return round(avg * 100, 1)
    return 0.0


def _build_radar_scores(
    team_idx: int,
    tactical_report: dict,
    match_report: dict,
    scored_report: dict,
) -> dict[str, float]:
    """Map scored team metrics to the 8 radar-chart axes (0–100)."""
    tk = f"team_{team_idx}"
    ts = scored_report.get("team_scores", {}).get(tk, {})
    meta = match_report.get("meta", {})
    formation_conf = (
        tactical_report.get("formation", {}).get(tk, {}).get("confidence", 0) * 100
    )
    passing = ts.get("passing_score")

    return {
        "kiem_soat_bong": float(ts.get("possession_score", meta.get(f"possession_team_{team_idx}", 0))),
        "doi_hinh":       _r2(formation_conf),
        "pressing":       float(ts.get("pressing_score", 0)),
        "ky_luat":        float(ts.get("turnovers_score", 0)),
        "toc_do":         50.0,
        "on_dinh":        float(ts.get("recoveries_score", 0)),
        "phong_thu":      float(passing) if passing is not None else 0.0,
        "do_rong":        float(ts.get("width_score", 0)),
    }


def _apply_radar_speed_norm(teams: list[dict]) -> None:
    """Scale avg_speed into 20–100 on the radar «Tốc độ» axis."""
    speeds = [float(t.get("metrics", {}).get("avg_speed", 0) or 0) for t in teams]
    lo, hi = min(speeds), max(speeds)
    for team, spd in zip(teams, speeds):
        radar = team.setdefault("metrics", {}).setdefault("radar_scores", {})
        if hi > lo:
            radar["toc_do"] = round(20 + (spd - lo) / (hi - lo) * 80, 1)
        else:
            radar["toc_do"] = 50.0


def _team_metrics(
    team_idx: int,
    tactical_report: dict,
    match_report: dict,
    scored_report: dict,
) -> dict[str, Any]:
    tk        = f"team_{team_idx}"
    meta      = match_report.get("meta", {})
    narrative = match_report.get("match_narrative_data", {})
    cau_truc  = match_report.get("cau_truc_doi_hinh",  {}).get(tk, {})
    mo_hinh   = match_report.get("mo_hinh_tran",       {}).get(tk, {})
    press_rec = match_report.get("press_and_recovery", {}).get(tk, {})

    compact_windows = (
        tactical_report.get("compact", {}).get(tk)
        or []
    )
    avg_compact = (
        sum(w.get("mean_area", 0) for w in compact_windows) / len(compact_windows)
        if compact_windows else 0.0
    )

    to_data   = tactical_report.get("turnovers", {}).get(tk, {})
    pass_data = (tactical_report.get("passing") or {}).get(tk, {})

    ppda = press_rec.get("ppda")
    pressing_intensity = _compute_pressing_intensity(
        team_idx, press_rec, scored_report,
    )

    def_line_m = float(
        mo_hinh.get("block_height_m")
        or narrative.get(f"def_line_avg_team_{team_idx}", 0)
    )
    def_line_pct = mo_hinh.get("block_height_pct")
    if def_line_pct is None:
        def_line_pct = _def_line_depth_pct(def_line_m)

    return {
        "possession":           float(meta.get(f"possession_team_{team_idx}", 0)),
        "compact_avg_m2":       round(avg_compact, 2),
        "compact_attacking_m2": cau_truc.get("compact_attacking_m2"),
        "compact_defending_m2": cau_truc.get("compact_defending_m2"),
        "compact_trend":        cau_truc.get("compact_trend", "stable"),
        "pressing_h1":          float(press_rec.get("pressing_h1", 0)),
        "pressing_h2":          float(press_rec.get("pressing_h2", 0)),
        "pressing_drop_pct":    float(press_rec.get("pressing_drop_pct", 0)),
        "pressing_intensity":   pressing_intensity,
        "ppda":                 float(ppda) if ppda is not None else None,
        "ppda_label":           press_rec.get("ppda_label"),
        "avg_speed":            float(narrative.get(f"speed_team_{team_idx}", 0)),
        "sprint_pct":           float(narrative.get(f"sprint_pct_team_{team_idx}", 0)),
        "defensive_line_height": def_line_m,
        "defensive_line_pct":    float(def_line_pct),
        "block_type":           mo_hinh.get("block_type", "mid_block"),
        "width":                float(narrative.get(f"width_avg_team_{team_idx}", 0)),
        "width_with_ball":      float(narrative.get(f"width_with_ball_team_{team_idx}", 0)),
        "width_without_ball":   float(narrative.get(f"width_without_ball_team_{team_idx}", 0)),
        "width_delta":          float(cau_truc.get("width_delta_m", 0)),
        "ball_recoveries":      int(press_rec.get("recoveries_total", 0)),
        "recoveries_opp_pct":   float(press_rec.get("recoveries_opp_pct", 0)),
        "turnovers_final_third":    int(to_data.get("total_turnovers_in_final_third", 0)),
        "high_risk_count":          int(to_data.get("high_risk_count",      0)),
        "dangerous_turnovers":      int(to_data.get("high_risk_count",      0)),
        "high_risk_rate_pct":       float(to_data.get("high_risk_rate_pct", 0)),
        "avg_distance_to_goal_m":   float(to_data.get("avg_distance_to_goal_m", 0)),
        "avg_transition_potential": float(to_data.get("avg_transition_potential", 0)),
        "high_intensity_runs":      int(
            tactical_report.get("high_intensity_runs", {}).get(tk, {}).get("total_runs", 0)
        ),
        "formation_adherence":      _r2(
            tactical_report.get("formation", {}).get(tk, {}).get("confidence", 0) * 100
        ),
        "forward_passes_pct":   float(
            pass_data.get("progressive_pass_pct", 0) if isinstance(pass_data, dict) else 0
        ),
        "radar_scores": _build_radar_scores(
            team_idx, tactical_report, match_report, scored_report,
        ),
    }


def _build_teams(
    tactical_report: dict,
    match_report: dict,
    scored_report: dict,
    llm_eval: dict | None,
    team_names: dict[int, str] | None = None,
) -> list[dict]:
    teams: list[dict] = []
    meta      = match_report.get("meta", {})
    danh_gia  = (llm_eval or {}).get("danh_gia_doi", {})
    cau_truc  = match_report.get("cau_truc_doi_hinh", {})
    insights  = match_report.get("insights", {})
    _names    = team_names or {}

    for team_idx in (1, 2):
        tk       = f"team_{team_idx}"
        llm_team = danh_gia.get(f"doi_{team_idx}", {})

        # Priority: detected scoreboard name > LLM name > fallback
        name = (
            _names.get(team_idx)
            or llm_team.get("ten")
            or f"Đội {team_idx}"
        )

        teams.append({
            "name":             name,
            "formation":        meta.get(f"formation_team_{team_idx}", "unknown"),
            "tactical_profile": llm_team.get("chien_thuat", ""),
            "insights":         insights.get(tk, []),
            "metrics":          _team_metrics(team_idx, tactical_report, match_report, scored_report),
        })
    _apply_def_line_pct_to_teams(teams, match_report)
    _apply_radar_speed_norm(teams)
    return teams


def _apply_def_line_pct_to_teams(teams: list[dict], match_report: dict | None = None) -> None:
    """Set ``defensive_line_pct`` on each team (0–100 % from own goal line)."""
    match_report = match_report or {}
    raw: list[float] = []
    for idx, team in enumerate(teams):
        team_idx = idx + 1
        m = team.get("metrics", {})
        raw.append(float(
            m.get("defensive_line_height")
            or match_report.get("mo_hinh_tran", {}).get(f"team_{team_idx}", {}).get("block_height_m", 0)
        ))

    use_shift = any(r < 0 for r in raw)
    floor = min(raw) if use_shift else 0.0

    for idx, team in enumerate(teams):
        metrics = team.setdefault("metrics", {})
        if metrics.get("defensive_line_pct") is not None and not use_shift:
            continue
        if use_shift:
            shifted = max(0.0, raw[idx] - floor)
            metrics["defensive_line_pct"] = round(shifted / _PITCH_LENGTH_M * 100, 1)
        else:
            metrics["defensive_line_pct"] = _def_line_depth_pct(raw[idx])


def enrich_teams_metrics(
    teams: list[dict],
    match_report: dict | None = None,
    scored_report: dict | None = None,
) -> list[dict]:
    """Back-fill metrics added after older jobs were saved."""
    match_report = match_report or {}
    scored_report = scored_report or {}
    press_rec_all = match_report.get("press_and_recovery", {})

    for idx, team in enumerate(teams):
        metrics = team.setdefault("metrics", {})
        if "pressing_intensity" in metrics:
            continue
        team_idx = idx + 1
        metrics["pressing_intensity"] = _compute_pressing_intensity(
            team_idx,
            press_rec_all.get(f"team_{team_idx}", {}),
            scored_report,
            metrics,
        )
        if "radar_scores" not in metrics:
            metrics["radar_scores"] = _build_radar_scores(
                team_idx, {}, match_report, scored_report,
            )
            if metrics.get("formation_adherence"):
                metrics["radar_scores"]["doi_hinh"] = float(metrics["formation_adherence"])
            if metrics.get("pressing_intensity"):
                metrics["radar_scores"]["pressing"] = float(metrics["pressing_intensity"])
    _apply_def_line_pct_to_teams(teams, match_report)
    _apply_radar_speed_norm(teams)
    return teams


def _compute_grade(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"


_ROLE_MAP = {
    "DEF": "DEF", "GK": "GK",
    "MID": "MID", "DM": "MID", "AM": "MID", "SS": "MID",
    "FWD": "FWD", "ST": "FWD",
}


def _build_position_lookup(match_report: dict) -> dict[int, str]:
    """Map track_id → role (DEF/MID/FWD) from formation lines in mo_hinh_tran."""
    lookup: dict[int, str] = {}
    for tk_val in match_report.get("mo_hinh_tran", {}).values():
        for raw_label, ids in tk_val.get("lines", {}).items():
            role = _ROLE_MAP.get(raw_label.upper(), "MID")
            for pid in ids:
                lookup[int(pid)] = role
    return lookup


def _build_players(
    match_report: dict,
    scored_report: dict,
    tactical_report: dict | None = None,
) -> list[dict]:
    players: list[dict] = []
    cau_thu    = match_report.get("cau_thu_then_chot", {})
    raw_scores = scored_report.get("player_scores", {})

    # Build position lookup from formation lines
    pos_lookup = _build_position_lookup(match_report)

    # Build per-player high-intensity run counts from tactical_report
    hi_runs_by_player: dict[int, int] = {}
    if tactical_report:
        hi_data = tactical_report.get("high_intensity_runs", {})
        for _tk, team_hi in hi_data.items():
            for ev in team_hi.get("run_events", []):
                pid = ev.get("track_id")
                if pid is not None:
                    hi_runs_by_player[int(pid)] = hi_runs_by_player.get(int(pid), 0) + 1

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
                "team":               team_id,
                "team_id":            team_id,
                "track_id":           tid,
                "squad_number":       squad_num,
                "display_name":       f"Cầu thủ {squad_num}",
                "position":           pos_lookup.get(tid, "MID"),
                "tags":               tags,
                "avg_speed":          round(float(raw.get("avg_speed_kmh", 0)), 1),
                "pressing":           round(float(raw.get("pressing_contrib", 0)) * 100, 1),
                "pressing_frames":    int(raw.get("pressing_frames", 0)),
                "discipline":         round(float(raw.get("def_line_std_m") or 0), 2),
                "coverage":           int(raw.get("active_frames", 0)),
                "high_intensity_runs": hi_runs_by_player.get(tid, 0),
                "creative_passes":    None,
                "width_contrib":      int(raw.get("flank_frames", 0)),
                "def_position":       raw.get("def_line_std_m"),
                "speed":              round(float(raw.get("avg_speed_kmh", 0)), 1),
                "activity":           int(raw.get("active_frames", 0)),
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


def _build_fallback_evaluation(
    match_report: dict,
    team_names: dict[int, str] | None = None,
) -> dict:
    """Structured evaluation from match_report when LLM is unavailable."""
    meta     = match_report.get("meta", {})
    narrative = match_report.get("match_narrative_data", {})
    insights  = match_report.get("insights", {})
    mo_hinh   = match_report.get("mo_hinh_tran", {})
    cau_truc  = match_report.get("cau_truc_doi_hinh", {})
    press_rec = match_report.get("press_and_recovery", {})
    _names    = team_names or {}

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
            "ten":          _names.get(team_idx) or f"Đội {team_idx}",
            "so_do":        formation,
            "chien_thuat":  (
                f"Sơ đồ {formation}, kiểm soát bóng {possession:.1f}%."
            ),
            "pressing":     (
                f"PPDA {pr.get('ppda'):.1f} ({pr.get('ppda_label', '')}). "
                f"Nửa đầu video: {pr.get('ppda_half1', '—')} | "
                f"Nửa sau: {pr.get('ppda_half2', '—')}"
                if pr.get("ppda") is not None
                else (
                    f"Proximity nửa đầu video: {h1:.2f} | nửa sau: {h2:.2f} "
                    f"({'giảm' if drop < 0 else 'tăng'} {abs(drop):.0f}%)"
                    if h1 else "Không có dữ liệu pressing."
                )
            ),
            "doi_hinh":     (
                cau_truc.get(tk, {}).get("compact_phase_comment")
                or f"Compact trend: {narrative.get(f'compact_trend_team_{team_idx}', 'stable')}"
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
            "khuyen_nghi_hlv": ["Cải thiện pressing ở nửa sau video", "Tăng cường kỷ luật vị trí hàng thủ"],
            "insights":     insights.get(tk, []),
        }

    dom      = meta.get("dominant_team")
    name_1   = _names.get(1) or "Đội 1"
    name_2   = _names.get(2) or "Đội 2"
    dom_name = _names.get(dom) or f"Đội {dom}" if dom else None
    return {
        "tong_quan_tran_dau": {
            "nhan_xet_chung": (
                f"Đội hình {meta.get('formation_team_1')} vs {meta.get('formation_team_2')}. "
                f"Kiểm soát bóng {meta.get('possession_team_1', 0):.1f}% — "
                f"{meta.get('possession_team_2', 0):.1f}%."
            ),
            "doi_noi_bat": dom,
            "ly_do": f"{dom_name} có compact và pressing tốt hơn." if dom_name else None,
        },
        "danh_gia_doi":     teams_out,
        "doi_1":            teams_out["doi_1"],
        "doi_2":            teams_out["doi_2"],
        "so_sanh_doi_dau": {
            "pressing": (
                f"Đầu: {press_rec.get('team_1', {}).get('pressing_h1', 0):.2f} / "
                f"{press_rec.get('team_2', {}).get('pressing_h1', 0):.2f} | "
                f"Cuối: {press_rec.get('team_1', {}).get('pressing_h2', 0):.2f} / "
                f"{press_rec.get('team_2', {}).get('pressing_h2', 0):.2f}"
            ),
            "kiem_soat_bong": (
                f"{name_1} {meta.get('possession_team_1', 0):.1f}% — "
                f"{name_2} {meta.get('possession_team_2', 0):.1f}%"
            ),
        },
        "ket_luan": (
            f"Trận đấu nghiêng về {dom_name} về mặt compact và pressing."
            if dom_name else "Hai đội cân bằng về chỉ số chiến thuật."
        ),
    }


def normalize_evaluation(
    llm_eval: dict | None,
    match_report: dict,
    team_names: dict[int, str] | None = None,
) -> dict:
    if llm_eval and llm_eval.get("danh_gia_doi"):
        out  = dict(llm_eval)
        _nm  = team_names or {}
        # Inject detected team names into the LLM evaluation
        for team_idx in (1, 2):
            doi_key = f"doi_{team_idx}"
            detected = _nm.get(team_idx)
            if detected:
                out.setdefault("danh_gia_doi", {}).setdefault(doi_key, {})["ten"] = detected
        out["doi_1"] = out.get("danh_gia_doi", {}).get("doi_1", {})
        out["doi_2"] = out.get("danh_gia_doi", {}).get("doi_2", {})
        if llm_eval.get("tong_quan_tran_dau") and not out.get("overview"):
            out["overview"] = llm_eval["tong_quan_tran_dau"]
        return out
    if llm_eval and not llm_eval.get("warning"):
        return llm_eval
    return _build_fallback_evaluation(match_report, team_names)


def adapt_api_result(
    match_report: dict,
    tactical_report: dict,
    scored_report: dict,
    charts: dict,
    passing_events: list[dict],
    llm_eval: dict | None,
    fps: float,
    team_names: dict[int, str] | None = None,
) -> dict:
    evaluation = normalize_evaluation(llm_eval, match_report, team_names)
    players    = _build_players(match_report, scored_report, tactical_report)
    return {
        "evaluation":      evaluation,
        "match_report":    match_report,
        "charts":          charts,
        "teams":           _build_teams(tactical_report, match_report, scored_report, llm_eval, team_names),
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

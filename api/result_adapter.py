"""
Normalise pipeline outputs into the shape expected by the React frontend.

The frontend expects top-level keys ``teams``, ``players``, ``evaluation``,
``charts``, ``timeline``, and ``notable_players`` in addition to the raw
``match_report``.
"""

from __future__ import annotations

from typing import Any


def _grade(score: float) -> str:
    for threshold, letter in ((80, "A"), (65, "B"), (50, "C"), (35, "D")):
        if score >= threshold:
            return letter
    return "F"


def _team_metrics(
    team_idx: int,
    tactical_report: dict,
    match_report: dict,
    scored_report: dict,
) -> dict[str, Any]:
    tk = f"team_{team_idx}"
    meta = match_report.get("meta", {})
    narrative = match_report.get("match_narrative_data", {})
    team_tr = match_report.get("team_report", {}).get(tk, {})
    scores = team_tr.get("scores", {})
    ts = scored_report.get("team_scores", {}).get(tk, {})

    compact_windows = (
        tactical_report.get("compact", {}).get(tk)
        or tactical_report.get("compact_score", {}).get(tk)
        or []
    )
    avg_compact = (
        sum(w.get("compact_score", 0) for w in compact_windows) / len(compact_windows)
        if compact_windows
        else 0.0
    )

    press_windows = tactical_report.get("pressing", {}).get("windows", [])
    team_press = [
        w.get("intensity", 0)
        for w in press_windows
        if w.get("pressing_team") == team_idx
    ]
    avg_press = sum(team_press) / len(team_press) if team_press else 0.0

    to_data = tactical_report.get("turnovers", {}).get(tk, {})
    pass_data = (tactical_report.get("passing") or {}).get(tk, {})

    return {
        "possession": float(meta.get(f"possession_team_{team_idx}", 0)),
        "compact_score": round(avg_compact, 2),
        "pressing_intensity": round(avg_press, 3),
        "formation_adherence": float(
            narrative.get(f"adherence_team_{team_idx}", scores.get("adherence", 0))
        ),
        "avg_speed": float(narrative.get(f"speed_team_{team_idx}", 0)),
        "sprint_pct": float(narrative.get(f"sprint_pct_team_{team_idx}", 0)),
        "defensive_line_height": float(
            narrative.get(f"def_line_avg_team_{team_idx}", 0)
        ),
        "width": float(narrative.get(f"width_avg_team_{team_idx}", 0)),
        "high_intensity_runs": int(narrative.get(f"high_runs_team_{team_idx}", 0)),
        "ball_recoveries": int(narrative.get(f"recoveries_team_{team_idx}", 0)),
        "dangerous_turnovers": int(
            to_data.get("total_turnovers_in_final_third", 0)
        ),
        "forward_passes_pct": float(
            pass_data.get("progressive_pass_pct", 0) if isinstance(pass_data, dict) else 0
        ),
        "overall_score": float(scores.get("overall", ts.get("overall_score", 50))),
        "stability": float(scores.get("stability", 50)),
        "discipline": float(scores.get("adherence", 50)),
        "width_normalized": min(
            100, float(narrative.get(f"width_avg_team_{team_idx}", 0)) / 0.55
        ),
        "avg_speed_normalized": min(
            100, float(narrative.get(f"speed_team_{team_idx}", 0)) / 0.3
        ),
        "defensive_score": float(scores.get("def_line", 50)),
        "ball_control": float(meta.get(f"possession_team_{team_idx}", 0)),
        "formation_adherence_pct": float(
            narrative.get(f"adherence_team_{team_idx}", scores.get("adherence", 0))
        ),
    }


def _build_teams(
    tactical_report: dict,
    match_report: dict,
    scored_report: dict,
    llm_eval: dict | None,
) -> list[dict]:
    teams: list[dict] = []
    meta = match_report.get("meta", {})
    danh_gia = (llm_eval or {}).get("danh_gia_doi", {})

    for team_idx in (1, 2):
        tk = f"team_{team_idx}"
        team_tr = match_report.get("team_report", {}).get(tk, {})
        scores = team_tr.get("scores", {})
        grades = team_tr.get("grades", {})
        llm_team = danh_gia.get(f"doi_{team_idx}", {})

        overall = float(scores.get("overall", 50))
        teams.append(
            {
                "name": llm_team.get("ten") or f"Đội {team_idx}",
                "formation": meta.get(f"formation_team_{team_idx}", "unknown"),
                "overall_score": overall,
                "grade": grades.get("overall") or _grade(overall),
                "tactical_profile": team_tr.get("tactical_profile", ""),
                "metrics": _team_metrics(team_idx, tactical_report, match_report, scored_report),
            }
        )
    return teams


def _build_players(
    match_report: dict,
    scored_report: dict,
) -> list[dict]:
    players: list[dict] = []
    player_report = match_report.get("player_report", {})
    raw_scores = scored_report.get("player_scores", {})

    for team_str, roster in player_report.items():
        team_id = int(team_str)
        team_raw = raw_scores.get(team_str, raw_scores.get(str(team_id), {}))

        for tid_str, info in roster.items():
            raw = team_raw.get(tid_str, {})
            players.append(
                {
                    "team": team_id,
                    "team_id": team_id,
                    "track_id": int(tid_str),
                    "position": info.get("role_in_line", "MID"),
                    "total_score": info.get("overall_score", raw.get("overall_score", 0)),
                    "grade": info.get("grade", _grade(info.get("overall_score", 0))),
                    "avg_speed": raw.get("speed_score", 0),
                    "pressing": raw.get("pressing_score", 0),
                    "discipline": raw.get("discipline_score", 0),
                    "coverage": raw.get("activity_score", 0),
                    "high_intensity_runs": raw.get("high_run_score", 0),
                    "creative_passes": raw.get("passing_involvement_score", 0),
                }
            )
    return players


def _build_timeline(passing_events: list[dict]) -> list[dict]:
    timeline: list[dict] = []
    for ev in passing_events:
        timeline.append(
            {
                "frame": ev.get("frame", 0),
                "team": max(0, int(ev.get("team", 1)) - 1),
                "type": "pass",
                "description": (
                    f"Chuyền bóng #{ev.get('passer_id')} → #{ev.get('receiver_id')}"
                ),
            }
        )
    return timeline


def _build_notable_players(llm_eval: dict | None, match_report: dict) -> dict:
    if llm_eval and llm_eval.get("danh_gia_cau_thu"):
        dc = llm_eval["danh_gia_cau_thu"]
        return {
            "team1_best": _player_card(dc.get("doi_1", {}).get("xuat_sac")),
            "team1_improve": _player_card(dc.get("doi_1", {}).get("can_cai_thien")),
            "team2_best": _player_card(dc.get("doi_2", {}).get("xuat_sac")),
            "team2_improve": _player_card(dc.get("doi_2", {}).get("can_cai_thien")),
        }

    notable: dict[str, Any] = {}
    for team_idx in (1, 2):
        tk = f"team_{team_idx}"
        tr = match_report.get("team_report", {}).get(tk, {})
        top = tr.get("top_players") or []
        weak = tr.get("weak_players") or []
        prefix = f"team{team_idx}"
        if top:
            notable[f"{prefix}_best"] = {
                "track_id": top[0],
                "reason": f"Cầu thủ #{top[0]} — điểm cao nhất đội {team_idx}",
            }
        if weak:
            notable[f"{prefix}_improve"] = {
                "track_id": weak[0],
                "reason": f"Cầu thủ #{weak[0]} — cần cải thiện hiệu suất",
                "recommendation": "Tăng cường pressing và kỷ luật vị trí.",
            }
    return notable


def _player_card(entry: dict | None) -> dict | None:
    if not entry:
        return None
    return {
        "track_id": entry.get("track_id"),
        "reason": entry.get("ly_do") or entry.get("reason"),
        "recommendation": entry.get("khuyen_nghi") or entry.get("recommendation"),
        "highlights": entry.get("chi_so_noi_bat"),
        "grade": entry.get("grade"),
        "position": entry.get("position"),
    }


def _build_fallback_evaluation(match_report: dict) -> dict:
    """Structured evaluation from match_report when LLM is unavailable."""
    meta = match_report.get("meta", {})
    narrative = match_report.get("match_narrative_data", {})
    teams_out: dict[str, Any] = {}

    for team_idx in (1, 2):
        tk = f"team_{team_idx}"
        tr = match_report.get("team_report", {}).get(tk, {})
        scores = tr.get("scores", {})
        grades = tr.get("grades", {})
        overall = float(scores.get("overall", 50))

        teams_out[f"doi_{team_idx}"] = {
            "ten": f"Đội {team_idx}",
            "diem_tong": overall,
            "diem_so_tong": overall,
            "xep_loai": grades.get("overall") or _grade(overall),
            "phong_cach": tr.get("tactical_profile", ""),
            "tactical_profile": tr.get("tactical_profile", ""),
            "so_do": meta.get(f"formation_team_{team_idx}", "unknown"),
            "chien_thuat": (
                f"Sơ đồ {meta.get(f'formation_team_{team_idx}', 'unknown')}, "
                f"kiểm soát bóng {meta.get(f'possession_team_{team_idx}', 0):.1f}%."
            ),
            "diem_manh": [
                f"Tốc độ trung bình {narrative.get(f'speed_team_{team_idx}', 0):.1f} km/h",
                f"Adherence đội hình {narrative.get(f'adherence_team_{team_idx}', 0):.1f}%",
            ],
            "diem_yeu": [
                f"Mất bóng vùng cuối sân: {narrative.get(f'turnovers_final_third_team_{team_idx}', 0)} lần",
            ],
            "nhan_xet_pressing": (
                f"Cường độ pressing H1/H2: "
                f"{narrative.get('pressing_intensity_half1', 0):.2f} / "
                f"{narrative.get('pressing_intensity_half2', 0):.2f}"
            ),
            "nhan_xet_doi_hinh": (
                f"Adherence {narrative.get(f'adherence_team_{team_idx}', 0):.1f}%, "
                f"compact trend: {narrative.get(f'compact_trend_team_{team_idx}', 'stable')}"
            ),
            "nhan_xet_toc_do": (
                f"{narrative.get(f'speed_team_{team_idx}', 0):.1f} km/h, "
                f"sprint {narrative.get(f'sprint_pct_team_{team_idx}', 0):.1f}%"
            ),
            "nhan_xet_hang_thu": (
                f"Hàng thủ trung bình {narrative.get(f'def_line_avg_team_{team_idx}', 0):.1f} m, "
                f"xu hướng {narrative.get(f'def_line_trend_team_{team_idx}', 'stable')}"
            ),
            "nhan_xet_do_rong": (
                f"Độ rộng TB {narrative.get(f'width_avg_team_{team_idx}', 0):.1f} m"
            ),
            "nhan_xet_van_dong": (
                f"{narrative.get(f'high_runs_team_{team_idx}', 0)} lần chạy cường độ cao"
            ),
            "nhan_xet_tranh_chap": (
                f"{narrative.get(f'recoveries_team_{team_idx}', 0)} lần cướp bóng"
            ),
            "nhan_xet_mat_bong": (
                f"{narrative.get(f'turnovers_final_third_team_{team_idx}', 0)} lần mất bóng ở phần sân đối phương"
            ),
            "nhan_xet_chuyen": (
                f"{narrative.get(f'total_passes_team_{team_idx}', 0) or 0} đường chuyền"
                if narrative.get(f"total_passes_team_{team_idx}") is not None
                else "Không đủ dữ liệu chuyền bóng để đánh giá."
            ),
        }

    dom = meta.get("dominant_team")
    dom_name = f"Đội {dom}" if dom else "Cân bằng"
    return {
        "tong_quan_tran_dau": {
            "nhan_xet_chung": (
                f"Đội hình {meta.get('formation_team_1')} vs {meta.get('formation_team_2')}. "
                f"Kiểm soát bóng {meta.get('possession_team_1', 0):.1f}% — "
                f"{meta.get('possession_team_2', 0):.1f}%."
            ),
            "doi_noi_bat": dom_name if dom else None,
            "ly_do": f"Đội {dom} có overall score cao hơn." if dom else None,
        },
        "danh_gia_doi": teams_out,
        "doi_1": teams_out["doi_1"],
        "doi_2": teams_out["doi_2"],
        "so_sanh_doi_dau": {
            "pressing": (
                f"Pressing H1: {narrative.get('pressing_intensity_half1', 0):.2f}, "
                f"H2: {narrative.get('pressing_intensity_half2', 0):.2f}"
            ),
            "kiem_soat_bong": (
                f"Đội 1 {meta.get('possession_team_1', 0):.1f}% — "
                f"Đội 2 {meta.get('possession_team_2', 0):.1f}%"
            ),
            "the_luc": (
                f"Tốc độ: {narrative.get('speed_team_1', 0):.1f} vs "
                f"{narrative.get('speed_team_2', 0):.1f} km/h"
            ),
        },
        "ket_luan": (
            f"Trận đấu nghiêng về {dom_name} về mặt chỉ số tổng thể."
            if dom
            else "Hai đội cân bằng về chỉ số tổng thể."
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
    return {
        "evaluation": evaluation,
        "match_report": match_report,
        "charts": charts,
        "teams": _build_teams(tactical_report, match_report, scored_report, llm_eval),
        "players": _build_players(match_report, scored_report),
        "timeline": _build_timeline(passing_events),
        "notable_players": _build_notable_players(llm_eval, match_report),
        "fps": fps,
    }


def build_streamlit_analysis_result(tactical_report: dict) -> dict[int, list[dict]]:
    """Build time-series samples for the Streamlit Plotly charts."""
    result: dict[int, list] = {1: [], 2: []}

    for team_idx in (1, 2):
        tk = f"team_{team_idx}"
        compact_w = (
            tactical_report.get("compact", {}).get(tk)
            or tactical_report.get("compact_score", {}).get(tk)
            or []
        )
        press_w = tactical_report.get("pressing", {}).get("windows", [])
        poss = tactical_report.get("possession", {})
        speeds = poss.get("avg_speed", {}).get(tk, {})
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
            result[team_idx].append(
                {
                    "frame": frame,
                    "formation": w.get("formation") or default_formation,
                    "compact": w.get("compact_score", 0),
                    "pressing": press_by_frame.get(frame, 0),
                    "avg_speed": team_speed,
                }
            )

        if not result[team_idx]:
            result[team_idx].append(
                {
                    "frame": 0,
                    "formation": default_formation,
                    "compact": 0,
                    "pressing": 0,
                    "avg_speed": team_speed,
                }
            )

    return result

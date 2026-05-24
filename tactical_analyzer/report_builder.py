"""
ReportBuilder
=============
Assembles a structured ``match_report`` dict from a ``ThresholdEngine``
scored report and a ``TacticalAnalyzer`` tactical report.

The output is JSON-serialisable and formatted for direct LLM consumption.
"""

from __future__ import annotations

from typing import Any

# ── constants ───────────────────────────────────────────────────────────────

_GRADE_MAP = ((80, "A"), (65, "B"), (50, "C"), (35, "D"))

_METRIC_LABELS: dict[str, str] = {
    "speed_score":               "Tốc độ di chuyển",
    "pressing_score":            "Cường độ pressing",
    "discipline_score":          "Kỷ luật chiến thuật",
    "activity_score":            "Độ phủ sóng sân",
    "def_positioning_score":     "Kỷ luật hàng thủ",
    "width_contrib_score":       "Khai thác biên dọc",
    "high_run_score":            "Cường độ chạy nước rút",
    "passing_involvement_score": "Tham gia xây dựng lối chơi",
}

_LINE_MAP: dict[str, str] = {
    "DEF": "DEF", "MID": "MID", "FWD": "FWD",
    "DM":  "MID", "AM":  "MID", "SS":  "MID",
}

# ── helpers ─────────────────────────────────────────────────────────────────

def _grade(score: float) -> str:
    for threshold, letter in _GRADE_MAP:
        if score >= threshold:
            return letter
    return "F"


def _r2(v: Any) -> float:
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return 0.0


# ── main class ───────────────────────────────────────────────────────────────

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
        ``"meta"``, ``"team_report"``, ``"player_report"``,
        ``"match_narrative_data"``.
        """
        if total_frames is None:
            total_frames = self._estimate_total_frames(tactical_report)

        return {
            "meta":                 self._build_meta(scored_report, tactical_report, total_frames),
            "team_report":          self._build_team_report(scored_report, tactical_report, total_frames),
            "player_report":        self._build_player_report(scored_report, tactical_report),
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

    # ── team_report ──────────────────────────────────────────────────────────

    def _build_team_report(
        self, scored: dict, tactical: dict, total_frames: int
    ) -> dict[str, Any]:
        return {
            f"team_{i}": self._build_one_team(scored, tactical, i, total_frames)
            for i in (1, 2)
        }

    def _build_one_team(
        self,
        scored: dict,
        tactical: dict,
        team_idx: int,
        total_frames: int,
    ) -> dict[str, Any]:
        tk  = f"team_{team_idx}"
        ts  = scored.get("team_scores", {}).get(tk, {})
        passing_raw = ts.get("passing_score")

        scores = {
            "overall":    _r2(ts.get("overall_score",   50.0)),
            "possession": _r2(ts.get("possession_score",50.0)),
            "compact":    _r2(ts.get("compact_score",   50.0)),
            "pressing":   _r2(ts.get("pressing_score",  50.0)),
            "adherence":  _r2(ts.get("adherence_score", 50.0)),
            "speed":      _r2(ts.get("speed_score",     50.0)),
            "stability":  _r2(ts.get("stability_score", 50.0)),
            "def_line":   _r2(ts.get("def_line_score",  50.0)),
            "width":      _r2(ts.get("width_score",     50.0)),
            "high_runs":  _r2(ts.get("high_runs_score", 50.0)),
            "recoveries": _r2(ts.get("recoveries_score",50.0)),
            "turnovers":  _r2(ts.get("turnovers_score", 50.0)),
            "passing":    _r2(passing_raw) if passing_raw is not None else None,
        }
        grades = {
            k: (_grade(v) if v is not None else None)
            for k, v in scores.items()
        }
        flags   = self._build_flags(scored, tactical, team_idx, total_frames)
        profile = self._build_tactical_profile(flags)
        top, weak = self._rank_players(
            scored.get("player_scores", {}).get(str(team_idx), {})
        )
        return {
            "scores":           scores,
            "grades":           grades,
            "flags":            flags,
            "tactical_profile": profile,
            "top_players":      top,
            "weak_players":     weak,
        }

    def _build_flags(
        self,
        scored: dict,
        tactical: dict,
        team_idx: int,
        total_frames: int,
    ) -> dict[str, Any]:
        tk     = f"team_{team_idx}"
        opp_tk = "team_2" if team_idx == 1 else "team_1"
        ts     = scored.get("team_scores", {}).get(tk,     {})
        opp_ts = scored.get("team_scores", {}).get(opp_tk, {})

        # high_press_half
        press_windows = self._pressing(tactical).get("windows", [])
        half_split    = total_frames // 2
        h1 = sum(
            1 for w in press_windows
            if w.get("pressing_team") == team_idx and w.get("window_start_frame", 0) <  half_split
        )
        h2 = sum(
            1 for w in press_windows
            if w.get("pressing_team") == team_idx and w.get("window_start_frame", 0) >= half_split
        )
        high_press_half: int | None = None
        if h1 > 0 or h2 > 0:
            high_press_half = 1 if h1 >= h2 else 2

        # formation_collapsed
        c_windows   = self._compact_windows(tactical, team_idx)
        broken_rate = (
            sum(1 for w in c_windows if w.get("formation_broken")) / len(c_windows)
            if c_windows else 0.0
        )

        # possession_dominant
        poss_pct = float(
            self._possession(tactical).get("possession", {}).get(f"team_{team_idx}", 0.0)
        )

        # def_line flags
        dl_data = self._def_line(tactical).get(tk, {})
        dl_avg  = float(dl_data.get("overall_avg_m") or 0.0)

        # width flags
        ww_data = self._team_width(tactical).get(tk, {})
        w_avg   = float(ww_data.get("overall_avg_m")       or 0.0)
        wball   = float(ww_data.get("width_with_ball")      or 0.0)
        wnoball = float(ww_data.get("width_without_ball")   or 0.0)

        # press_recovery
        rec_data     = tactical.get("ball_recoveries", {}).get(tk, {})
        rec_total    = int(rec_data.get("total_recoveries", 0))
        rec_zones    = rec_data.get("recoveries_by_zone", {})
        opp_half_rec = int(rec_zones.get("opp_half", 0)) + int(rec_zones.get("final_third", 0))
        press_recovery = rec_total > 0 and (opp_half_rec / rec_total) > 0.4

        # risky_buildup
        to_data       = tactical.get("turnovers", {}).get(tk, {})
        risky_buildup = float(to_data.get("dangerous_rate_pct", 0.0)) > 40.0

        # progressive_team
        pass_data        = (tactical.get("passing") or {}).get(tk, {})
        prog_pct         = float(pass_data.get("progressive_pass_pct", 0.0)) if isinstance(pass_data, dict) else 0.0
        progressive_team = prog_pct > 35.0

        # high_runner_team
        hr_data          = tactical.get("high_intensity_runs", {})
        my_runs          = int(hr_data.get(tk, {}).get("total_runs", 0))
        opp_runs         = int(hr_data.get(opp_tk, {}).get("total_runs", 0))
        high_runner_team = my_runs > opp_runs

        return {
            "high_press_half":           high_press_half,
            "formation_collapsed":       broken_rate > 0.20,
            "speed_dominant":            ts.get("speed_score", 50.0) > opp_ts.get("speed_score", 50.0),
            "possession_dominant":       poss_pct > 55.0,
            "deep_defending":            bool(dl_avg > 0 and dl_avg < 8.0),
            "high_pressing_block":       bool(dl_avg > 14.0),
            "wide_play_style":           bool(w_avg > 45.0),
            "width_expands_with_ball":   (wball - wnoball) > 5.0,
            "press_recovery":            press_recovery,
            "risky_buildup":             risky_buildup,
            "progressive_team":          progressive_team,
            "high_runner_team":          high_runner_team,
        }

    @staticmethod
    def _build_tactical_profile(flags: dict) -> str:
        high        = flags.get("high_pressing_block", False)
        deep        = flags.get("deep_defending",       False)
        wide        = flags.get("wide_play_style",       False)
        runner      = flags.get("high_runner_team",      False)
        progressive = flags.get("progressive_team",      False)
        risky       = flags.get("risky_buildup",         False)
        pressing_rec = flags.get("press_recovery",       False)

        if high and wide:
            base = "High-press, wide attacking style"
        elif high:
            base = "High-press, narrow compact style"
        elif deep and wide:
            base = "Deep block, wide defensive shape"
        elif deep:
            base = "Deep block, narrow defensive shape"
        else:
            base = "Balanced mid-block style"

        suffixes: list[str] = []
        if runner and progressive:
            suffixes.append("direct attacking")
        if risky:
            suffixes.append("risky in possession")
        if pressing_rec:
            suffixes.append("aggressive recovery")

        return base + (", " + ", ".join(suffixes) if suffixes else "")

    @staticmethod
    def _rank_players(player_scores: dict) -> tuple[list[int], list[int]]:
        if not player_scores:
            return [], []
        ranked = sorted(
            player_scores.items(),
            key=lambda kv: kv[1].get("overall_score", 0.0),
            reverse=True,
        )
        n       = len(ranked)
        top_n   = min(3, n)
        weak_n  = min(3, n)
        top_ids  = [int(k) for k, _ in ranked[:top_n]]
        weak_ids = [int(k) for k, _ in ranked[n - weak_n:]]
        weak_ids = [pid for pid in weak_ids if pid not in top_ids] if n > top_n else weak_ids
        return top_ids, weak_ids

    # ── player_report ────────────────────────────────────────────────────────

    def _build_player_report(
        self, scored: dict, tactical: dict
    ) -> dict[str, dict[str, Any]]:
        role_lookup: dict[int, str] = {}
        for team_idx in (1, 2):
            for raw_label, ids in self._formation_team(tactical, team_idx).get("lines", {}).items():
                mapped = _LINE_MAP.get(raw_label, "MID")
                for pid in ids:
                    role_lookup[int(pid)] = mapped

        result: dict[str, dict[str, Any]] = {}
        for team_str, players in scored.get("player_scores", {}).items():
            if not players:
                result[team_str] = {}
                continue
            team_out: dict[str, Any] = {}
            for tid_str, pscores in players.items():
                overall    = _r2(pscores.get("overall_score", 0.0))
                strengths  = [lbl for key, lbl in _METRIC_LABELS.items() if pscores.get(key, 0.0) >= 70.0]
                weaknesses = [lbl for key, lbl in _METRIC_LABELS.items() if pscores.get(key, 0.0) <  40.0]
                team_out[tid_str] = {
                    "overall_score": overall,
                    "grade":         _grade(overall),
                    "role_in_line":  role_lookup.get(int(tid_str), "MID"),
                    "strengths":     strengths,
                    "weaknesses":    weaknesses,
                }
            result[team_str] = team_out
        return result

    # ── match_narrative_data ──────────────────────────────────────────────────

    def _build_narrative(self, tactical: dict) -> dict[str, Any]:
        press    = self._pressing(tactical)
        half_sum = press.get("half_summary", {})
        poss     = self._possession(tactical)
        spd      = poss.get("avg_speed", {})
        zones    = poss.get("speed_zones", {})

        def _hr(tk: str) -> dict:
            return tactical.get("high_intensity_runs", {}).get(tk, {})

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
            "adherence_team_1":                _r2(self._formation_team(tactical, 1).get("adherence_score", 0.0)),
            "adherence_team_2":                _r2(self._formation_team(tactical, 2).get("adherence_score", 0.0)),
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
            "high_runs_team_1":                int(_hr("team_1").get("total_runs", 0)),
            "high_runs_team_2":                int(_hr("team_2").get("total_runs", 0)),
            "high_runs_fwd_team_1":            int(_hr("team_1").get("runs_per_role", {}).get("FWD", 0)),
            "high_runs_fwd_team_2":            int(_hr("team_2").get("runs_per_role", {}).get("FWD", 0)),
            "recoveries_team_1":               int(_rec("team_1").get("total_recoveries", 0)),
            "recoveries_team_2":               int(_rec("team_2").get("total_recoveries", 0)),
            "opp_half_recovery_pct_team_1":    _opp_half_pct("team_1"),
            "opp_half_recovery_pct_team_2":    _opp_half_pct("team_2"),
            "turnovers_final_third_team_1":    int(_to("team_1").get("total_turnovers_in_final_third", 0)),
            "turnovers_final_third_team_2":    int(_to("team_2").get("total_turnovers_in_final_third", 0)),
            "dangerous_rate_team_1":           _r2(_to("team_1").get("dangerous_rate_pct", 0.0)),
            "dangerous_rate_team_2":           _r2(_to("team_2").get("dangerous_rate_pct", 0.0)),
            "total_passes_team_1":             int(p1.get("total_passes", 0)) if isinstance(p1, dict) else None,
            "total_passes_team_2":             int(p2.get("total_passes", 0)) if isinstance(p2, dict) else None,
            "progressive_pct_team_1":          _r2(p1.get("progressive_pass_pct", 0.0)) if isinstance(p1, dict) else None,
            "progressive_pct_team_2":          _r2(p2.get("progressive_pass_pct", 0.0)) if isinstance(p2, dict) else None,
        }

    def _compact_trend(self, tactical: dict, team_idx: int) -> str:
        windows = self._compact_windows(tactical, team_idx)
        n = len(windows)
        if n < 2:
            return "stable"
        half  = max(n // 2, 1)
        avg1  = sum(w["compact_score"] for w in windows[:half]) / half
        avg2  = sum(w["compact_score"] for w in windows[half:]) / max(n - half, 1)
        diff  = avg2 - avg1
        if diff < -1.0:
            return "improving"
        if diff >  1.0:
            return "declining"
        return "stable"

"""
ThresholdEngine
===============
Converts a ``TacticalAnalyzer.analyze()`` report into normalised 0-100
scores using **match-relative percentile thresholds**.

Coordinate system (matches ViewTransformer / TacticalAnalyzer):
    pos[0] = x = along pitch LENGTH  (0 → _VISIBLE_LENGTH ≈ 23.32 m)
    pos[1] = y = across pitch WIDTH  (0 → _PITCH_WIDTH = 68 m)

All scores are clipped to [0, 100] and JSON-serialisable.
"""

from __future__ import annotations

import numpy as np
from typing import Any

# FIFA-standard pitch dimensions
_PITCH_LENGTH  = 105.0   # full pitch length (metres)
_PITCH_WIDTH   =  68.0   # full pitch width  (metres)
# Visible portion of pitch length in the default camera calibration
_VISIBLE_LENGTH = 23.32
# Flank thresholds on the WIDTH axis (y = 0-68 m):
# players within outer 25 % of pitch width count as "flank contribution"
_FLANK_LEFT  = 17.0   # 25 % of 68 m  (near far touchline)
_FLANK_RIGHT = 51.0   # 75 % of 68 m  (near camera touchline)

# ── tiny utilities ──────────────────────────────────────────────────────────

def _clip100(v: float) -> float:
    return float(np.clip(v, 0.0, 100.0))


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * float(np.clip(t, 0.0, 1.0))


def _pct_rank(value: float, distribution: np.ndarray) -> float:
    if len(distribution) <= 1:
        return 0.5
    return float(np.mean(distribution <= value))


def _safe_percentile(arr: np.ndarray, qs: list[float], fallback: float) -> list[float]:
    if len(arr) < 2:
        return [arr[0] if len(arr) == 1 else fallback] * len(qs)
    return [float(np.percentile(arr, q)) for q in qs]


class ThresholdEngine:
    """Score a TacticalAnalyzer report with match-relative percentile thresholds.

    Parameters
    ----------
    fps : int
        Video frame rate (default 24).
    R_pressing : float
        Pressing radius in metres (default 5.5).
        Must match the value used in ``TacticalAnalyzer``.
        pitch_length : float
        Actual pitch length covered by pos[0].  Must match the value passed
        to ``TacticalAnalyzer``.  Used only as a cap / normaliser in the
        defensive-line scoring function.
    """

    def __init__(
        self,
        fps: int = 24,
        R_pressing: float = 5.5,
        pitch_length: float = _PITCH_LENGTH,
    ) -> None:
        self.fps          = fps
        self.R_pressing   = R_pressing
        self.pitch_length = float(pitch_length)

    # ── public API ──────────────────────────────────────────────────────────

    def compute(self, tactical_report: dict, tracks: dict) -> dict[str, Any]:
        """Compute all scores from the tactical report.

        Parameters
        ----------
        tactical_report : dict
            Output of ``TacticalAnalyzer.analyze()``.
        tracks : dict
            Full pipeline tracks dict (for per-player scoring).

        Returns
        -------
        JSON-serialisable scored_report dict.
        """
        thresholds   = self._build_thresholds(tactical_report)
        team_scores  = self._score_teams(tactical_report, thresholds)
        player_scores = self._score_players(tactical_report, tracks)
        match_summary = self._build_match_summary(tactical_report, team_scores)
        return {
            "thresholds":    thresholds,
            "team_scores":   team_scores,
            "player_scores": player_scores,
            "match_summary": match_summary,
        }

    # ── threshold builder ───────────────────────────────────────────────────

    def _build_thresholds(self, rpt: dict) -> dict[str, Any]:
        # compact — read mean_area (compact_score alias removed)
        compact_vals = [
            w["mean_area"]
            for tk in ("team_1", "team_2")
            for w in rpt.get("compact", {}).get(tk, [])
            if "mean_area" in w
        ]
        p25, p50, p75 = _safe_percentile(
            np.array(compact_vals, dtype=float), [25, 50, 75], fallback=10.0
        )

        # pressing
        press_vals = [w["intensity"] for w in rpt.get("pressing", {}).get("windows", [])]
        pr33, pr67 = _safe_percentile(
            np.array(press_vals, dtype=float), [33, 67], fallback=2.0
        )

        # def_line
        dl_vals = [
            w["def_line_height_m"]
            for tk in ("team_1", "team_2")
            for w in rpt.get("def_line", {}).get(tk, {}).get("windows", [])
        ]
        dl33, dl67 = _safe_percentile(
            np.array(dl_vals, dtype=float), [33, 67], fallback=self.pitch_length / 2
        )

        # team_width
        ww_vals = [
            w["width_m"]
            for tk in ("team_1", "team_2")
            for w in rpt.get("team_width", {}).get(tk, {}).get("windows", [])
        ]
        ww33, ww67 = _safe_percentile(
            np.array(ww_vals, dtype=float), [33, 67], fallback=_PITCH_WIDTH / 2
        )

        # ball_recoveries
        rec_vals = [
            float(rpt.get("ball_recoveries", {}).get(tk, {}).get("total_recoveries", 0))
            for tk in ("team_1", "team_2")
        ]
        rec33, rec67 = _safe_percentile(
            np.array(rec_vals, dtype=float), [33, 67], fallback=5.0
        )

        # turnovers
        to_vals = [
            float(rpt.get("turnovers", {}).get(tk, {}).get("total_turnovers_in_final_third", 0))
            for tk in ("team_1", "team_2")
        ]
        to33, to67 = _safe_percentile(
            np.array(to_vals, dtype=float), [33, 67], fallback=2.0
        )

        return {
            "compact":    {"p25": round(p25,   4), "p50": round(p50,   4), "p75": round(p75,   4)},
            "pressing":   {"p33": round(pr33,  4), "p67": round(pr67,  4)},
            "def_line":   {"p33": round(dl33,  4), "p67": round(dl67,  4)},
            "width":      {"p33": round(ww33,  4), "p67": round(ww67,  4)},
            "recoveries": {"p33": round(rec33, 4), "p67": round(rec67, 4)},
            "turnovers":  {"p33": round(to33,  4), "p67": round(to67,  4)},
        }

    # ── per-value scoring helpers ────────────────────────────────────────────

    @staticmethod
    def _score_compact_val(val: float, p25: float, p50: float, p75: float) -> float:
        """compact_score (metres, lower = better) → 0-100."""
        if p25 <= 0 or p50 <= p25 or p75 <= p50:
            return 50.0
        if val <= p25:
            return _clip100(_lerp(85.0, 100.0, 1.0 - val / p25))
        if val <= p50:
            return _clip100(_lerp(65.0, 84.0,  1.0 - (val - p25) / (p50 - p25)))
        if val <= p75:
            return _clip100(_lerp(45.0, 64.0,  1.0 - (val - p50) / (p75 - p50)))
        excess = (val - p75) / max(p75, 1e-6)
        return _clip100(_lerp(0.0, 44.0, 1.0 - excess))

    @staticmethod
    def _score_pressing_val(val: float, p33: float, p67: float, max_val: float) -> float:
        """pressing intensity (higher = better) → 0-100."""
        if p33 <= 0 or p67 <= p33:
            return 50.0
        if val >= p67:
            return _clip100(_lerp(75.0, 100.0, (val - p67) / max(max_val - p67, 1e-6)))
        if val >= p33:
            return _clip100(_lerp(40.0, 74.0,  (val - p33) / (p67 - p33)))
        return _clip100(_lerp(0.0, 39.0, val / p33))

    @staticmethod
    def _score_def_line_val(
        val: float, p33: float, p67: float, pitch_length: float = _PITCH_LENGTH
    ) -> float:
        """def_line_height (metres) → 0-100, style-neutral."""
        if p33 <= 0 or p67 <= p33:
            return 50.0
        if val >= p67:
            return _clip100(70.0 + ((val - p67) / max(pitch_length - p67, 1e-6)) * 30.0)
        if val >= p33:
            return _clip100(_lerp(50.0, 69.0, (val - p33) / (p67 - p33)))
        return _clip100(30.0 + (val / max(p33, 1e-6)) * 19.0)

    @staticmethod
    def _score_width_val(val: float, p33: float, p67: float) -> float:
        """team_width (metres) → 0-100, style-neutral."""
        if p33 <= 0 or p67 <= p33:
            return 50.0
        if val >= p67:
            return _clip100(70.0 + ((val - p67) / max(_PITCH_WIDTH - p67, 1e-6)) * 30.0)
        if val >= p33:
            return _clip100(_lerp(50.0, 69.0, (val - p33) / (p67 - p33)))
        return _clip100(30.0 + (val / max(p33, 1e-6)) * 19.0)

    @staticmethod
    def _score_higher_better(val: float, p33: float, p67: float, max_val: float) -> float:
        """Generic higher-is-better 3-band scorer (runs, recoveries)."""
        if p33 <= 0 or p67 <= p33:
            return 50.0
        if val >= p67:
            return _clip100(_lerp(75.0, 100.0, (val - p67) / max(max_val - p67, 1e-6)))
        if val >= p33:
            return _clip100(_lerp(45.0, 74.0,  (val - p33) / (p67 - p33)))
        return _clip100(_lerp(0.0, 44.0, val / p33))

    @staticmethod
    def _score_turnovers_val(
        val: float, p33: float, p67: float, high_risk_rate: float
    ) -> float:
        """turnovers_in_final_third (lower = better) → 0-100.

        Contextual penalty: high_risk_rate (proportion of turnovers classified
        as High-Risk by transition-potential analysis) is used as a continuous
        modifier instead of a fixed 40% hard threshold.  A turnover set with
        100% high-risk rate loses up to 15 points; 0% loses none.
        """
        if p33 <= 0 or p67 <= p33:
            base = 50.0
        elif val <= p33:
            base = _clip100(_lerp(75.0, 100.0, 1.0 - val / max(p33, 1e-6)))
        elif val <= p67:
            base = _clip100(_lerp(45.0, 74.0,  1.0 - (val - p33) / (p67 - p33)))
        else:
            excess = (val - p67) / max(p67, 1e-6)
            base = _clip100(_lerp(0.0, 44.0, 1.0 - excess))
        risk_penalty = 15.0 * float(np.clip(high_risk_rate / 100.0, 0.0, 1.0))
        return _clip100(base - risk_penalty)

    # ── team scoring ─────────────────────────────────────────────────────────

    def _score_teams(self, rpt: dict, thresholds: dict) -> dict[str, Any]:
        p = thresholds

        all_press_windows = rpt.get("pressing", {}).get("windows", [])
        max_press         = max((w["intensity"] for w in all_press_windows), default=1.0)

        result: dict[str, Any] = {}

        for team_idx, tk in enumerate(("team_1", "team_2"), start=1):

            # ── 1. Possession ──────────────────────────────────────────────
            poss_pct        = float(rpt.get("possession", {}).get("possession", {}).get(tk, 50.0))
            possession_score = _clip100(poss_pct)

            # ── 2. Compact (raw hull area m² — no 0-100 normalisation) ───────
            c_windows = rpt.get("compact", {}).get(tk, [])
            if c_windows:
                compact_avg_m2 = float(np.mean([
                    w["mean_area"]
                    for w in c_windows
                    if "mean_area" in w
                ]))
            else:
                compact_avg_m2 = None

            # ── 3. Pressing ─────────────────────────────────────────────────
            press_windows_team = [
                w for w in all_press_windows if w.get("pressing_team") == team_idx
            ]
            if press_windows_team:
                pressing_score = float(np.mean([
                    self._score_pressing_val(
                        w["intensity"],
                        p["pressing"]["p33"], p["pressing"]["p67"], max_press,
                    )
                    for w in press_windows_team
                ]))
            else:
                pressing_score = 50.0

            # ── 4. Def-line ─────────────────────────────────────────────────
            dl_windows = rpt.get("def_line", {}).get(tk, {}).get("windows", [])
            if dl_windows:
                def_line_score = float(np.mean([
                    self._score_def_line_val(
                        w["def_line_height_m"],
                        p["def_line"]["p33"], p["def_line"]["p67"],
                        self.pitch_length,
                    )
                    for w in dl_windows
                ]))
            else:
                def_line_score = 50.0

            # ── 5. Team width ───────────────────────────────────────────────
            w_windows = rpt.get("team_width", {}).get(tk, {}).get("windows", [])
            if w_windows:
                width_base = float(np.mean([
                    self._score_width_val(
                        w["width_m"],
                        p["width"]["p33"], p["width"]["p67"],
                    )
                    for w in w_windows
                ]))
            else:
                width_base = 50.0
            tw_data     = rpt.get("team_width", {}).get(tk, {})
            wball       = float(tw_data.get("width_with_ball",    0.0))
            wnoball     = float(tw_data.get("width_without_ball", 0.0))
            width_score = _clip100(width_base + (5.0 if wball - wnoball > 5.0 else 0.0))

            # ── 6. Ball recoveries ──────────────────────────────────────────
            rec_data  = rpt.get("ball_recoveries", {}).get(tk, {})
            rec_total = float(rec_data.get("total_recoveries", 0))
            max_rec   = max(
                float(rpt.get("ball_recoveries", {}).get(t, {}).get("total_recoveries", 0))
                for t in ("team_1", "team_2")
            ) or 1.0
            rec_base  = self._score_higher_better(
                rec_total, p["recoveries"]["p33"], p["recoveries"]["p67"], max_rec
            )
            rec_zones = rec_data.get("recoveries_by_zone", {})
            opp_rec   = float(rec_zones.get("opp_half", 0)) + float(rec_zones.get("final_third", 0))
            rec_score = _clip100(
                rec_base + (5.0 if rec_total > 0 and opp_rec / rec_total > 0.4 else 0.0)
            )

            # ── 7. Turnovers in final third ─────────────────────────────────
            to_data         = rpt.get("turnovers", {}).get(tk, {})
            to_total        = float(to_data.get("total_turnovers_in_final_third", 0))
            to_high_risk    = float(to_data.get("high_risk_rate_pct", 0.0))
            turnovers_score = self._score_turnovers_val(
                to_total, p["turnovers"]["p33"], p["turnovers"]["p67"], to_high_risk
            )

            # ── 8. Progressive passing ──────────────────────────────────────
            # Score = progressive_pass_pct mapped linearly to [0, 100].
            # Also factors in network_density (diversity of connections) with
            # a 20 % weight so a team that passes to many different partners
            # gets rewarded alongside forward-pass volume.
            # Formula:  0.80 × prog_pct  +  0.20 × (density × 100)
            # No artificial floor — a team with 0 % progressive passes
            # genuinely scores 0 in this dimension.
            passing_data = (rpt.get("passing") or {}).get(tk)
            if passing_data and isinstance(passing_data, dict):
                prog_pct      = float(passing_data.get("progressive_pass_pct", 0.0))
                net_density   = float(passing_data.get("network_density", 0.0))
                passing_score: float | None = _clip100(
                    0.80 * prog_pct + 0.20 * net_density * 100.0
                )
            else:
                passing_score = None

            result[tk] = {
                "possession_score":  round(possession_score,  2),
                "compact_avg_m2":    round(compact_avg_m2, 2) if compact_avg_m2 is not None else None,
                "pressing_score":    round(pressing_score,    2),
                "def_line_score":    round(def_line_score,    2),
                "width_score":       round(width_score,       2),
                "recoveries_score":  round(rec_score,         2),
                "turnovers_score":   round(turnovers_score,   2),
                "passing_score":     round(passing_score, 2) if passing_score is not None else None,
            }

        return result

    # ── player scoring ────────────────────────────────────────────────────────

    def _score_players(self, rpt: dict, tracks: dict) -> dict[str, Any]:
        """Compute 8 player-level scores.

        Players are extracted directly from ``tracks["players"]``.
        """
        frames       = tracks.get("players", [])
        total_frames = len(frames)
        if total_frames == 0:
            return {}

        R = self.R_pressing

        # Formation line lookup
        player_line: dict[int, str] = {}
        _label_map = {"DEF": "DEF", "MID": "MID", "FWD": "FWD",
                      "DM": "MID", "AM": "MID", "SS": "MID"}
        for tk in ("team_1", "team_2"):
            lines = rpt.get("formation", {}).get(tk, {}).get("lines", {})
            for raw_label, ids in lines.items():
                mapped = _label_map.get(raw_label, "MID")
                for pid in ids:
                    player_line[int(pid)] = mapped

        # DEF player sets per team
        def_ids_by_team: dict[int, set[int]] = {1: set(), 2: set()}
        for pid, line in player_line.items():
            if line != "DEF":
                continue
            for fi in range(min(total_frames, 50)):
                if pid in frames[fi]:
                    t = frames[fi][pid].get("team")
                    if t in (1, 2):
                        def_ids_by_team[t].add(pid)
                    break

        # Team def-line mean depth (pos[0] = along pitch length) per frame
        team_def_x: dict[int, list[float]] = {1: [float("nan")] * total_frames,
                                               2: [float("nan")] * total_frames}
        for fi, frame in enumerate(frames):
            for t in (1, 2):
                def_xs = [
                    float(frame[tid]["position_transformed"][0])  # depth direction
                    for tid in def_ids_by_team[t]
                    if tid in frame and frame[tid].get("position_transformed") is not None
                ]
                if def_xs:
                    team_def_x[t][fi] = float(np.mean(def_xs))

        # Per-team speed distributions (kept for future use; players use raw km/h)
        spd_lists: dict[int, list[float]] = {1: [], 2: []}
        for fi, frame in enumerate(frames):
            for tid, info in frame.items():
                t   = info.get("team")
                spd = info.get("speed")
                if t in (1, 2) and spd is not None:
                    spd_lists[t].append(float(spd))

        player_data: dict[int, dict] = {}

        for fi, frame in enumerate(frames):
            # Find ball carrier per team for pressing calc
            carrier_pos: dict[int, Any] = {}
            for info in frame.values():
                t = info.get("team")
                if t in (1, 2) and info.get("has_ball"):
                    carrier_pos[t] = info.get("position_transformed")

            for tid, info in frame.items():
                t = info.get("team")
                if t not in (1, 2):
                    continue
                if tid not in player_data:
                    player_data[tid] = {
                        "team":              t,
                        "by_frame":          {},
                        "pressing_frames":   0,
                        "opp_has_ball_frames": 0,
                    }
                pos = info.get("position_transformed")
                spd = info.get("speed")
                player_data[tid]["by_frame"][fi] = {"pos": pos, "spd": spd}

                opp = 3 - t
                if carrier_pos.get(opp) is not None:
                    player_data[tid]["opp_has_ball_frames"] += 1
                    if pos is not None:
                        dx   = float(pos[0]) - float(carrier_pos[opp][0])
                        dy   = float(pos[1]) - float(carrier_pos[opp][1])
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist <= R:
                            player_data[tid]["pressing_frames"] += 1

        result: dict[str, dict] = {"1": {}, "2": {}}

        for tid, data in player_data.items():
            team_id  = data["team"]
            by_frame = data["by_frame"]
            active_positions = [v["pos"] for v in by_frame.values() if v["pos"] is not None]
            active_count     = len(active_positions)
            active_frames    = active_count

            spds    = [v["spd"] for v in by_frame.values() if v["spd"] is not None]
            avg_spd = float(np.mean(spds)) if spds else 0.0

            contrib = data["pressing_frames"] / max(data["opp_has_ball_frames"], 1)

            line = player_line.get(tid, "MID")
            def_line_std_m: float | None = None
            if line == "DEF":
                def_x_ref  = team_def_x[team_id]
                deviations = [
                    v["pos"][0] - def_x_ref[fi]
                    for fi, v in by_frame.items()
                    if v["pos"] is not None
                    and fi < len(def_x_ref)
                    and not np.isnan(def_x_ref[fi])
                ]
                if deviations:
                    def_line_std_m = round(float(np.std(deviations)), 2)

            flank_frames = 0
            if line in ("MID", "FWD") and active_count > 0:
                flank_frames = sum(
                    1 for x, y in active_positions
                    if y <= _FLANK_LEFT or y >= _FLANK_RIGHT
                )

            result[str(team_id)][str(tid)] = {
                "avg_speed_kmh":         round(avg_spd, 2),
                "pressing_frames":       data["pressing_frames"],
                "opp_has_ball_frames":   data["opp_has_ball_frames"],
                "pressing_contrib":      round(contrib, 4),
                "active_frames":         active_frames,
                "def_line_std_m":        def_line_std_m,
                "flank_frames":          flank_frames,
                # Legacy keys for ranked lists in ReportBuilder
                "pressing_score":        round(contrib * 100.0, 2),
                "width_contrib_score":   round(
                    (flank_frames / active_count * 100.0) if active_count else 0.0, 2
                ),
                "def_positioning_score": round(
                    max(0.0, 100.0 - (def_line_std_m or 0.0) * 10.0), 2
                ) if line == "DEF" else None,
                "activity_score":        round(active_frames / max(total_frames, 1) * 100.0, 2),
            }

        return result

    # ── match summary ─────────────────────────────────────────────────────────

    def _build_match_summary(self, rpt: dict, team_scores: dict) -> dict[str, Any]:
        poss   = rpt.get("possession", {}).get("possession", {})
        t1_pct = float(poss.get("team_1", 50.0))
        t2_pct = float(poss.get("team_2", 50.0))

        # Dominant team: pressing + recoveries (compact is raw m², not scored)
        def _composite(tk: str) -> float:
            ts = team_scores.get(tk, {})
            return (
                ts.get("pressing_score",  50.0) * 0.50
                + ts.get("recoveries_score", 50.0) * 0.50
            )
        t1_comp = _composite("team_1")
        t2_comp = _composite("team_2")
        if abs(t1_comp - t2_comp) < 5.0:
            dominant_team: int | None = None
        else:
            dominant_team = 1 if t1_comp > t2_comp else 2

        possession_balance = (
            "balanced" if abs(t1_pct - t2_pct) < 10.0
            else ("dominant" if max(t1_pct, t2_pct) > 60.0 else "balanced")
        )

        press_windows = rpt.get("pressing", {}).get("windows", [])
        if press_windows:
            high_ratio      = sum(1 for w in press_windows if w.get("high_press")) / len(press_windows)
            avg_press_level = "high" if high_ratio >= 0.5 else ("medium" if high_ratio >= 0.2 else "low")
        else:
            avg_press_level = "low"

        # press_recovery_team
        def _opp_half_rec(tk: str) -> int:
            z = rpt.get("ball_recoveries", {}).get(tk, {}).get("recoveries_by_zone", {})
            return int(z.get("opp_half", 0)) + int(z.get("final_third", 0))

        opr1, opr2 = _opp_half_rec("team_1"), _opp_half_rec("team_2")
        press_recovery_team: int | None = None if opr1 == opr2 else (1 if opr1 > opr2 else 2)

        # risky_team
        # Use high_risk_count for risky_team: team with more contextually
        # dangerous turnovers (opponent had fast transition potential)
        to1 = int(rpt.get("turnovers", {}).get("team_1", {}).get("high_risk_count", 0))
        to2 = int(rpt.get("turnovers", {}).get("team_2", {}).get("high_risk_count", 0))
        risky_team: int | None = None if to1 == to2 else (1 if to1 > to2 else 2)

        return {
            "dominant_team":        dominant_team,
            "possession_balance":   possession_balance,
            "avg_press_level":      avg_press_level,
            "formation_team_1":     rpt.get("formation",  {}).get("team_1", {}).get("detected_formation", "unknown"),
            "formation_team_2":     rpt.get("formation",  {}).get("team_2", {}).get("detected_formation", "unknown"),
            "def_style_team_1":     rpt.get("def_line",   {}).get("team_1", {}).get("dominant_block",     "mid_block"),
            "def_style_team_2":     rpt.get("def_line",   {}).get("team_2", {}).get("dominant_block",     "mid_block"),
            "width_style_team_1":   rpt.get("team_width", {}).get("team_1", {}).get("dominant_style",     "medium"),
            "width_style_team_2":   rpt.get("team_width", {}).get("team_2", {}).get("dominant_style",     "medium"),
            "press_recovery_team":  press_recovery_team,
            "risky_team":           risky_team,
        }

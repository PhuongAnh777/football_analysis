"""
TacticalAnalyzer
================
Extracts raw tactical metrics from a pipeline ``tracks`` dict.

All values are in **metres** (pitch: x 0-68 m, y 0-23.32 m) and
**km/h** for speed, and are JSON-serialisable.

Methods
-------
Public (original 4):
    compact_score            → report["compact"]
    pressing_intensity       → report["pressing"]
    formation_adherence      → report["formation"]
    possession_stats         → report["possession"]

Private (methods 5-10):
    _defensive_line_height   → report["def_line"]
    _team_width              → report["team_width"]
    _high_intensity_runs     → report["high_intensity_runs"]
    _ball_recoveries         → report["ball_recoveries"]
    _turnovers_final_third   → report["turnovers"]
    _passing_stats           → report["passing"]
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

_PITCH_WIDTH = 68.0
_PITCH_DEPTH = 23.32


class TacticalAnalyzer:
    """Compute tactical metrics from pipeline tracks.

    Parameters
    ----------
    fps : int
        Video frame rate (default 24).
    window_sec : int
        Analysis window length in seconds (default 30).
    R_pressing : float
        Pressing radius in metres (default 8.0).
    """

    def __init__(
        self,
        fps: int = 24,
        window_sec: int = 30,
        R_pressing: float = 8.0,
    ) -> None:
        self.fps           = fps
        self.window_sec    = window_sec
        self.window_frames = fps * window_sec
        self.R_pressing    = R_pressing

    # ── public orchestrator ──────────────────────────────────────────────────

    def analyze(
        self,
        tracks: dict,
        team_ball_control: list[int],
        passing_events: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Run all analyses and return a single JSON-serialisable dict.

        Parameters
        ----------
        tracks : dict
            Full pipeline tracks dict (team + ball assigned, transformed).
        team_ball_control : list[int]
            Per-frame ball-control label: 1, 2, or 0.
        passing_events : list[dict] | None
            Optional list of detected pass events.

        Returns
        -------
        dict with keys: compact, pressing, formation, possession,
        def_line, team_width, high_intensity_runs, ball_recoveries,
        turnovers, passing.
        """
        report: dict[str, Any] = {}
        report["compact"]             = self.compact_score(tracks)
        report["pressing"]            = self.pressing_intensity(tracks, team_ball_control)
        report["formation"]           = self.formation_adherence(tracks)
        report["possession"]          = self.possession_stats(tracks, team_ball_control)
        report["def_line"]            = self._defensive_line_height(tracks)
        report["team_width"]          = self._team_width(tracks)
        report["high_intensity_runs"] = self._high_intensity_runs(tracks)
        report["ball_recoveries"]     = self._ball_recoveries(tracks, team_ball_control)
        report["turnovers"]           = self._turnovers_final_third(tracks, team_ball_control)
        report["passing"]             = (
            self._passing_stats(passing_events) if passing_events else None
        )
        return report

    # ── 1. compact_score ─────────────────────────────────────────────────────

    def compact_score(self, tracks: dict) -> dict[str, Any]:
        """Mean distance of players from their team centroid per 30-s window (metres).

        Lower compact_score = tighter, more organised formation.
        formation_broken = True when compact_score > 15 m.

        Returns
        -------
        {"team_1": [{"window_start_frame": int, "compact_score": float,
                     "formation_broken": bool}],
         "team_2": [...]}
        """
        frames = tracks["players"]
        n      = len(frames)
        W      = self.window_frames
        result: dict[str, list] = {"team_1": [], "team_2": []}

        for w_start in range(0, n, W):
            w_end = min(w_start + W, n)

            for team_idx in (1, 2):
                xs: list[float] = []
                ys: list[float] = []

                for fi in range(w_start, w_end):
                    for info in frames[fi].values():
                        if info.get("team") != team_idx:
                            continue
                        pos = info.get("position_transformed")
                        if pos is None:
                            continue
                        xs.append(float(pos[0]))
                        ys.append(float(pos[1]))

                if len(xs) < 2:
                    continue

                cx    = float(np.mean(xs))
                cy    = float(np.mean(ys))
                dists = np.sqrt((np.array(xs) - cx) ** 2 + (np.array(ys) - cy) ** 2)
                score = float(np.mean(dists))

                result[f"team_{team_idx}"].append({
                    "window_start_frame": w_start,
                    "compact_score":      round(score, 4),
                    "formation_broken":   score > 15.0,
                })

        return result

    # ── 2. pressing_intensity ────────────────────────────────────────────────

    def pressing_intensity(
        self, tracks: dict, team_ball_control: list[int]
    ) -> dict[str, Any]:
        """Count pressing-team players within R_pressing metres of ball carrier.

        Per 30-s window, reports the team with higher mean press count.
        high_press = True when intensity >= 2.0 players within radius.

        Returns
        -------
        {"windows": [{"window_start_frame": int, "pressing_team": int,
                      "intensity": float, "high_press": bool}],
         "half_summary": {"half_1": {"mean_intensity": float, "peak_intensity": float},
                          "half_2": {...}}}
        """
        frames = tracks["players"]
        n      = len(team_ball_control)
        W      = self.window_frames
        R      = self.R_pressing
        windows: list[dict] = []

        for w_start in range(0, n, W):
            w_end = min(w_start + W, n)
            intensities: dict[int, list[float]] = {1: [], 2: []}

            for fi in range(w_start, w_end):
                if fi >= len(frames):
                    continue
                ball_team = team_ball_control[fi]
                if ball_team not in (1, 2):
                    continue

                # Find ball carrier position
                carrier_pos = None
                for info in frames[fi].values():
                    if info.get("team") == ball_team and info.get("has_ball"):
                        carrier_pos = info.get("position_transformed")
                        break
                if carrier_pos is None:
                    continue

                # Count pressing team (opponent) players near carrier
                press_team = 3 - ball_team
                count = 0.0
                for info in frames[fi].values():
                    if info.get("team") != press_team:
                        continue
                    pos = info.get("position_transformed")
                    if pos is None:
                        continue
                    dx   = float(pos[0]) - float(carrier_pos[0])
                    dy   = float(pos[1]) - float(carrier_pos[1])
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist <= R:
                        count += 1.0
                intensities[press_team].append(count)

            mean1 = float(np.mean(intensities[1])) if intensities[1] else 0.0
            mean2 = float(np.mean(intensities[2])) if intensities[2] else 0.0

            if not intensities[1] and not intensities[2]:
                continue

            pt         = 1 if mean1 >= mean2 else 2
            intensity  = mean1 if pt == 1 else mean2
            windows.append({
                "window_start_frame": w_start,
                "pressing_team":      pt,
                "intensity":          round(intensity, 4),
                "high_press":         intensity >= 2.0,
            })

        half_split = n // 2

        def _summarise(ws: list[dict]) -> dict:
            if not ws:
                return {"mean_intensity": 0.0, "peak_intensity": 0.0}
            vals = [w["intensity"] for w in ws]
            return {
                "mean_intensity": round(float(np.mean(vals)), 4),
                "peak_intensity": round(float(np.max(vals)),  4),
            }

        return {
            "windows":      windows,
            "half_summary": {
                "half_1": _summarise([w for w in windows if w["window_start_frame"] <  half_split]),
                "half_2": _summarise([w for w in windows if w["window_start_frame"] >= half_split]),
            },
        }

    # ── 3. formation_adherence ───────────────────────────────────────────────

    def formation_adherence(self, tracks: dict) -> dict[str, Any]:
        """Cluster players into DEF/MID/FWD lines and compute adherence score.

        adherence_score (0-1): higher = players stay closer to their median
        y-position, i.e. better positional discipline.

        Returns
        -------
        {"team_1": {"detected_formation": str, "confidence": float,
                    "adherence_score": float,
                    "lines": {"DEF": [ids], "MID": [ids], "FWD": [ids]}},
         "team_2": {...}}
        """
        frames = tracks["players"]
        n      = len(frames)
        result: dict[str, Any] = {}

        # Accumulate per-player y and x positions
        player_data: dict[int, dict] = {}
        for frame in frames:
            for tid, info in frame.items():
                team = info.get("team")
                if team not in (1, 2):
                    continue
                pos = info.get("position_transformed")
                if pos is None:
                    continue
                if tid not in player_data:
                    player_data[tid] = {"team": team, "ys": [], "xs": []}
                player_data[tid]["ys"].append(float(pos[1]))
                player_data[tid]["xs"].append(float(pos[0]))

        min_frames = max(1, int(n * 0.05))   # player must appear in ≥5 % of frames

        for team_idx in (1, 2):
            team_players = {
                tid: d for tid, d in player_data.items()
                if d["team"] == team_idx and len(d["ys"]) >= min_frames
            }

            if not team_players:
                result[f"team_{team_idx}"] = {
                    "detected_formation": "unknown",
                    "confidence":         0.0,
                    "adherence_score":    0.5,
                    "lines": {"DEF": [], "MID": [], "FWD": []},
                }
                continue

            median_ys = {tid: float(np.median(d["ys"])) for tid, d in team_players.items()}
            sorted_p  = sorted(median_ys.items(), key=lambda x: x[1])
            n_p       = len(sorted_p)

            # Line counts: prefer 4-X-Y patterns
            if n_p >= 10:
                def_n = 4
                fwd_n = 3 if n_p >= 11 else 2
            elif n_p >= 7:
                def_n = 3
                fwd_n = 2
            else:
                def_n = max(1, n_p // 3)
                fwd_n = max(1, n_p // 4)
            mid_n = max(1, n_p - def_n - fwd_n)

            lines = {
                "DEF": [int(sorted_p[i][0]) for i in range(def_n)],
                "MID": [int(sorted_p[i][0]) for i in range(def_n, def_n + mid_n)],
                "FWD": [int(sorted_p[i][0]) for i in range(def_n + mid_n, n_p)],
            }
            formation = f"{def_n}-{mid_n}-{fwd_n}"

            # adherence_score: 1 - normalised mean std of y deviations (10 m → 0)
            stds = [
                float(np.std(team_players[tid]["ys"]))
                for tid in team_players
            ]
            mean_std       = float(np.mean(stds)) if stds else 5.0
            adherence_score = float(np.clip(1.0 - mean_std / 10.0, 0.0, 1.0))

            result[f"team_{team_idx}"] = {
                "detected_formation": formation,
                "confidence":         round(adherence_score, 4),
                "adherence_score":    round(adherence_score, 4),
                "lines":              lines,
            }

        return result

    # ── 4. possession_stats ──────────────────────────────────────────────────

    def possession_stats(
        self, tracks: dict, team_ball_control: list[int]
    ) -> dict[str, Any]:
        """Possession %, average speed, and speed-zone distribution.

        Speed zones (km/h):
            walking   < 7
            jogging   7-14
            running   14-20
            sprinting >= 20

        Returns
        -------
        {"possession": {"team_1": float, "team_2": float},
         "avg_speed":  {"team_1": {"overall": float, "per_window": [float]},
                        "team_2": {...}},
         "speed_zones": {"team_1": {"walking": float, "jogging": float,
                                    "running": float, "sprinting": float},
                         "team_2": {...}}}
        """
        frames = tracks["players"]
        n_ctrl = len(team_ball_control)
        n_fr   = len(frames)
        W      = self.window_frames

        # Possession
        t1_ctrl = sum(1 for c in team_ball_control if c == 1)
        t2_ctrl = sum(1 for c in team_ball_control if c == 2)
        total   = t1_ctrl + t2_ctrl or 1
        t1_pct  = t1_ctrl / total * 100.0
        t2_pct  = t2_ctrl / total * 100.0

        # Speed accumulation
        all_speeds:  dict[int, list[float]] = {1: [], 2: []}
        per_window:  dict[int, list[float]] = {1: [], 2: []}

        for w_start in range(0, max(n_ctrl, n_fr), W):
            w_end  = min(w_start + W, n_fr)
            w_spds: dict[int, list[float]] = {1: [], 2: []}

            for fi in range(w_start, w_end):
                if fi >= n_fr:
                    continue
                for info in frames[fi].values():
                    team = info.get("team")
                    if team not in (1, 2):
                        continue
                    spd = info.get("speed")
                    if spd is not None:
                        v = float(spd)
                        w_spds[team].append(v)
                        all_speeds[team].append(v)

            for t in (1, 2):
                per_window[t].append(
                    round(float(np.mean(w_spds[t])) if w_spds[t] else 0.0, 4)
                )

        # Speed zones
        speed_zones: dict[str, Any] = {}
        for t in (1, 2):
            vals = np.array(all_speeds[t], dtype=float) if all_speeds[t] else np.zeros(1)
            n_v  = max(len(vals), 1)
            speed_zones[f"team_{t}"] = {
                "walking":   round(float(np.sum(vals < 7)                       / n_v * 100), 2),
                "jogging":   round(float(np.sum((vals >= 7)  & (vals < 14))     / n_v * 100), 2),
                "running":   round(float(np.sum((vals >= 14) & (vals < 20))     / n_v * 100), 2),
                "sprinting": round(float(np.sum(vals >= 20)                     / n_v * 100), 2),
            }

        return {
            "possession": {
                "team_1": round(t1_pct, 2),
                "team_2": round(t2_pct, 2),
            },
            "avg_speed": {
                "team_1": {
                    "overall":    round(float(np.mean(all_speeds[1])) if all_speeds[1] else 0.0, 4),
                    "per_window": per_window[1],
                },
                "team_2": {
                    "overall":    round(float(np.mean(all_speeds[2])) if all_speeds[2] else 0.0, 4),
                    "per_window": per_window[2],
                },
            },
            "speed_zones": speed_zones,
        }

    # ── 5. defensive_line_height ─────────────────────────────────────────────

    def _defensive_line_height(self, tracks: dict) -> dict[str, Any]:
        """Average y-position of the defensive line per 30-s window (metres).

        y-axis: 0 = own goal line, 23.32 m = opponent goal line.
        high_block >= 14 m | mid_block 8-14 m | low_block < 8 m.

        Returns
        -------
        dict mapping "team_1" | "team_2" → {
            "windows": [{"window_start_frame": int,
                         "def_line_height_m": float, "block_type": str}],
            "overall_avg_m": float,
            "half_1_avg_m": float,
            "half_2_avg_m": float,
            "trend": "dropping"|"rising"|"stable",
            "dominant_block": str
        }
        """
        frames = tracks["players"]
        n      = len(frames)
        W      = self.window_frames
        half   = n // 2

        # DEF player sets from formation
        try:
            formation = self.formation_adherence(tracks)
        except Exception:
            formation = {}

        def_ids: dict[int, set[int]] = {1: set(), 2: set()}
        for team_idx in (1, 2):
            line_ids = (
                (formation.get(f"team_{team_idx}") or {})
                .get("lines", {})
                .get("DEF", [])
            )
            def_ids[team_idx] = {int(i) for i in line_ids}

        result: dict[str, Any] = {}
        for team_idx in (1, 2):
            windows: list[dict] = []

            for w_start in range(0, n, W):
                w_end  = min(w_start + W, n)
                frame_heights: list[float] = []

                for fi in range(w_start, w_end):
                    frame = frames[fi]

                    # Collect DEF player y-positions
                    def_ys: list[float] = []

                    if def_ids[team_idx]:
                        for tid in def_ids[team_idx]:
                            info = frame.get(tid, {})
                            if info.get("team") != team_idx:
                                continue
                            pos = info.get("position_transformed")
                            if pos is not None:
                                def_ys.append(float(pos[1]))
                    else:
                        # Fallback: 4 players with smallest y
                        team_ys = [
                            (float(info["position_transformed"][1]), tid)
                            for tid, info in frame.items()
                            if info.get("team") == team_idx
                            and info.get("position_transformed") is not None
                        ]
                        team_ys.sort()
                        def_ys = [y for y, _ in team_ys[:4]]

                    if len(def_ys) >= 3:
                        frame_heights.append(float(np.mean(def_ys)))

                if not frame_heights:
                    continue

                avg_h = float(np.mean(frame_heights))
                block = (
                    "high_block" if avg_h >= 14.0 else
                    "low_block"  if avg_h <   8.0 else
                    "mid_block"
                )
                windows.append({
                    "window_start_frame": w_start,
                    "def_line_height_m":  round(avg_h, 4),
                    "block_type":         block,
                })

            if not windows:
                result[f"team_{team_idx}"] = {
                    "windows":        [],
                    "overall_avg_m":  0.0,
                    "half_1_avg_m":   0.0,
                    "half_2_avg_m":   0.0,
                    "trend":          "stable",
                    "dominant_block": "mid_block",
                }
                continue

            all_h  = [w["def_line_height_m"] for w in windows]
            h1_h   = [w["def_line_height_m"] for w in windows if w["window_start_frame"] <  half]
            h2_h   = [w["def_line_height_m"] for w in windows if w["window_start_frame"] >= half]
            avg1   = float(np.mean(h1_h)) if h1_h else float(np.mean(all_h))
            avg2   = float(np.mean(h2_h)) if h2_h else avg1
            diff   = avg2 - avg1
            trend  = "dropping" if diff < -1.5 else ("rising" if diff > 1.5 else "stable")

            counts = {"high_block": 0, "mid_block": 0, "low_block": 0}
            for w in windows:
                counts[w["block_type"]] += 1
            dominant = max(counts, key=lambda k: counts[k])

            result[f"team_{team_idx}"] = {
                "windows":        windows,
                "overall_avg_m":  round(float(np.mean(all_h)), 4),
                "half_1_avg_m":   round(avg1, 4),
                "half_2_avg_m":   round(avg2, 4),
                "trend":          trend,
                "dominant_block": dominant,
            }

        return result

    # ── 6. team_width ────────────────────────────────────────────────────────

    def _team_width(self, tracks: dict) -> dict[str, Any]:
        """Horizontal spread of a team per 30-s window (metres, x-axis 0-68 m).

        wide >= 45 m | medium 30-45 m | narrow < 30 m.
        Also computes mean width when team has/doesn't have the ball.

        Returns
        -------
        dict mapping "team_1" | "team_2" → {
            "windows": [{"window_start_frame": int,
                         "width_m": float, "style": str}],
            "overall_avg_m": float,
            "half_1_avg_m": float,
            "half_2_avg_m": float,
            "width_with_ball": float,
            "width_without_ball": float,
            "trend": "expanding"|"contracting"|"stable",
            "dominant_style": str
        }
        """
        frames = tracks["players"]
        n      = len(frames)
        W      = self.window_frames
        half   = n // 2

        result: dict[str, Any] = {}
        for team_idx in (1, 2):
            windows: list[dict]    = []
            widths_ball:    list[float] = []
            widths_no_ball: list[float] = []

            for w_start in range(0, n, W):
                w_end = min(w_start + W, n)
                frame_widths: list[float] = []

                for fi in range(w_start, w_end):
                    frame = frames[fi]
                    xs = [
                        float(info["position_transformed"][0])
                        for info in frame.values()
                        if info.get("team") == team_idx
                        and info.get("position_transformed") is not None
                    ]
                    if len(xs) < 5:
                        continue
                    w_frame = float(np.max(xs) - np.min(xs))
                    frame_widths.append(w_frame)

                    # Ball possession context
                    team_has_ball = any(
                        info.get("has_ball") and info.get("team") == team_idx
                        for info in frame.values()
                    )
                    if team_has_ball:
                        widths_ball.append(w_frame)
                    else:
                        widths_no_ball.append(w_frame)

                if not frame_widths:
                    continue

                avg_w = float(np.mean(frame_widths))
                style = (
                    "wide"   if avg_w >= 45.0 else
                    "narrow" if avg_w <  30.0 else
                    "medium"
                )
                windows.append({
                    "window_start_frame": w_start,
                    "width_m":            round(avg_w, 4),
                    "style":              style,
                })

            if not windows:
                result[f"team_{team_idx}"] = {
                    "windows":            [],
                    "overall_avg_m":      0.0,
                    "half_1_avg_m":       0.0,
                    "half_2_avg_m":       0.0,
                    "width_with_ball":    0.0,
                    "width_without_ball": 0.0,
                    "trend":              "stable",
                    "dominant_style":     "medium",
                }
                continue

            all_w  = [w["width_m"] for w in windows]
            h1_w   = [w["width_m"] for w in windows if w["window_start_frame"] <  half]
            h2_w   = [w["width_m"] for w in windows if w["window_start_frame"] >= half]
            avg1   = float(np.mean(h1_w)) if h1_w else float(np.mean(all_w))
            avg2   = float(np.mean(h2_w)) if h2_w else avg1
            diff   = avg2 - avg1
            trend  = "expanding" if diff > 3.0 else ("contracting" if diff < -3.0 else "stable")

            counts = {"wide": 0, "medium": 0, "narrow": 0}
            for w in windows:
                counts[w["style"]] += 1
            dominant = max(counts, key=lambda k: counts[k])

            result[f"team_{team_idx}"] = {
                "windows":            windows,
                "overall_avg_m":      round(float(np.mean(all_w)), 4),
                "half_1_avg_m":       round(avg1, 4),
                "half_2_avg_m":       round(avg2, 4),
                "width_with_ball":    round(float(np.mean(widths_ball))    if widths_ball    else 0.0, 4),
                "width_without_ball": round(float(np.mean(widths_no_ball)) if widths_no_ball else 0.0, 4),
                "trend":              trend,
                "dominant_style":     dominant,
            }

        return result

    # ── 7. high_intensity_runs ────────────────────────────────────────────────

    def _high_intensity_runs(self, tracks: dict) -> dict[str, Any]:
        """Count high-intensity run events (speed > 20 km/h for >= 3 consecutive frames).

        Returns
        -------
        dict mapping "team_1" | "team_2" → {
            "total_runs": int,
            "runs_per_role": {"DEF": int, "MID": int, "FWD": int},
            "runs_half_1": int,
            "runs_half_2": int,
            "avg_peak_speed_kmh": float,
            "top_runner_id": int | None,
            "run_events": [{"track_id": int, "role": str,
                            "start_frame": int, "end_frame": int,
                            "peak_speed_kmh": float}]
        }
        """
        SPEED_THR  = 20.0
        MIN_FRAMES = 3
        _LABEL_MAP = {"DEF": "DEF", "MID": "MID", "FWD": "FWD",
                      "DM":  "MID", "AM":  "MID", "SS":  "MID"}

        frames       = tracks["players"]
        total_frames = len(frames)
        half_split   = total_frames // 2

        # Role lookup from formation
        try:
            formation_data = self.formation_adherence(tracks)
        except Exception:
            formation_data = {}

        player_role: dict[int, str] = {}
        for team_idx in (1, 2):
            lines = (
                (formation_data.get(f"team_{team_idx}") or {})
                .get("lines", {})
            )
            for raw_lbl, ids in lines.items():
                mapped = _LABEL_MAP.get(raw_lbl, "MID")
                for pid in ids:
                    player_role[int(pid)] = mapped

        # Fallback role by median-y ranking
        if not player_role:
            for team_idx in (1, 2):
                median_ys: dict[int, list[float]] = {}
                for frame in frames:
                    for tid, info in frame.items():
                        if info.get("team") != team_idx:
                            continue
                        pos = info.get("position_transformed")
                        if pos is None:
                            continue
                        median_ys.setdefault(int(tid), []).append(float(pos[1]))
                ranked = sorted(
                    {tid: float(np.median(ys)) for tid, ys in median_ys.items()}.items(),
                    key=lambda x: x[1],
                )
                n_p = len(ranked)
                def_n = max(1, n_p // 4)
                fwd_n = max(1, n_p // 4)
                for i, (tid, _) in enumerate(ranked):
                    if i < def_n:
                        player_role[tid] = "DEF"
                    elif i >= n_p - fwd_n:
                        player_role[tid] = "FWD"
                    else:
                        player_role[tid] = "MID"

        # Per-player frame speeds
        player_meta: dict[int, dict] = {}
        for fi, frame in enumerate(frames):
            for tid, info in frame.items():
                team = info.get("team")
                if team not in (1, 2):
                    continue
                if tid not in player_meta:
                    player_meta[tid] = {"team": team, "speeds": [None] * total_frames}
                spd = info.get("speed")
                if spd is not None:
                    player_meta[tid]["speeds"][fi] = float(spd)

        # Detect run events
        all_runs: list[dict] = []
        for tid, meta in player_meta.items():
            role   = player_role.get(tid, "MID")
            speeds = meta["speeds"]
            in_run = False
            run_start = 0
            peak = 0.0
            for fi, spd in enumerate(speeds):
                above = spd is not None and spd > SPEED_THR
                if above:
                    if not in_run:
                        in_run    = True
                        run_start = fi
                        peak      = spd
                    else:
                        peak = max(peak, spd)
                else:
                    if in_run:
                        if fi - run_start >= MIN_FRAMES:
                            all_runs.append({
                                "track_id":       int(tid),
                                "role":           role,
                                "start_frame":    run_start,
                                "end_frame":      fi,
                                "peak_speed_kmh": round(peak, 4),
                            })
                        in_run = False
                        peak   = 0.0
            if in_run and (total_frames - run_start) >= MIN_FRAMES:
                all_runs.append({
                    "track_id":       int(tid),
                    "role":           role,
                    "start_frame":    run_start,
                    "end_frame":      total_frames,
                    "peak_speed_kmh": round(peak, 4),
                })

        result: dict[str, Any] = {}
        for team_idx in (1, 2):
            t_runs = [
                r for r in all_runs
                if player_meta.get(r["track_id"], {}).get("team") == team_idx
            ]
            rpr: dict[str, int] = {"DEF": 0, "MID": 0, "FWD": 0}
            for r in t_runs:
                rpr[r["role"]] = rpr.get(r["role"], 0) + 1

            peaks      = [r["peak_speed_kmh"] for r in t_runs]
            runner_ctr = Counter(r["track_id"] for r in t_runs)
            top_runner = runner_ctr.most_common(1)[0][0] if runner_ctr else None

            result[f"team_{team_idx}"] = {
                "total_runs":         len(t_runs),
                "runs_per_role":      rpr,
                "runs_half_1":        sum(1 for r in t_runs if r["start_frame"] <  half_split),
                "runs_half_2":        sum(1 for r in t_runs if r["start_frame"] >= half_split),
                "avg_peak_speed_kmh": round(float(np.mean(peaks)) if peaks else 0.0, 4),
                "top_runner_id":      int(top_runner) if top_runner is not None else None,
                "run_events":         t_runs,
            }
        return result

    # ── 8. ball_recoveries ────────────────────────────────────────────────────

    def _ball_recoveries(
        self, tracks: dict, team_ball_control: list[int]
    ) -> dict[str, Any]:
        """Count ball recovery events per team.

        A recovery = team_ball_control transitions to T for >= 3 consecutive
        frames from a state where T did not hold the ball.

        Zone classification (y-axis metres):
            own_half    y < 11.66
            opp_half    11.66 <= y < 15.5
            final_third y >= 15.5

        Returns
        -------
        dict mapping "team_1" | "team_2" → {
            "total_recoveries": int,
            "recoveries_by_zone": {"own_half": int, "opp_half": int, "final_third": int},
            "recoveries_half_1": int,
            "recoveries_half_2": int,
            "recovery_rate_per100frames": float
        }
        """
        MIN_CTRL     = 3
        PITCH_MID    = 11.66
        FINAL_3RD    = 15.5
        ctrl         = team_ball_control
        total_frames = len(ctrl)
        half_split   = total_frames // 2
        n_pframes    = len(tracks["players"])

        result: dict[str, Any] = {}
        for team_idx in (1, 2):
            events: list[dict] = []
            i = 1
            while i < total_frames:
                if ctrl[i] == team_idx and ctrl[i - 1] != team_idx:
                    j = i
                    while j < total_frames and ctrl[j] == team_idx:
                        j += 1
                    if j - i >= MIN_CTRL:
                        pos = None
                        for fi in range(i, min(i + 5, n_pframes)):
                            for info in tracks["players"][fi].values():
                                if (info.get("team") == team_idx
                                        and info.get("has_ball")
                                        and info.get("position_transformed") is not None):
                                    pos = info["position_transformed"]
                                    break
                            if pos is not None:
                                break

                        zone = "own_half"
                        if pos is not None:
                            y = float(pos[1])
                            if y >= FINAL_3RD:
                                zone = "final_third"
                            elif y >= PITCH_MID:
                                zone = "opp_half"

                        events.append({"frame": i, "zone": zone})
                    i = j
                else:
                    i += 1

            by_zone = {"own_half": 0, "opp_half": 0, "final_third": 0}
            for ev in events:
                by_zone[ev["zone"]] += 1

            total = len(events)
            result[f"team_{team_idx}"] = {
                "total_recoveries":           total,
                "recoveries_by_zone":         by_zone,
                "recoveries_half_1":          sum(1 for e in events if e["frame"] <  half_split),
                "recoveries_half_2":          sum(1 for e in events if e["frame"] >= half_split),
                "recovery_rate_per100frames": round(total / max(total_frames, 1) * 100, 4),
            }
        return result

    # ── 9. turnovers_final_third ──────────────────────────────────────────────

    def _turnovers_final_third(
        self, tracks: dict, team_ball_control: list[int]
    ) -> dict[str, Any]:
        """Count turnovers occurring in the attacking final third.

        A turnover is counted when possession leaves team T and the opponent
        holds the ball for >= 5 consecutive frames (noise filter).
        Only turnovers where T's ball carrier is inside their attacking
        final third are counted.

        Attacking direction inferred from FWD players' median y:
            y_increasing → final_third: y >= 15.5 m
            y_decreasing → final_third: y <= 7.82 m

        Returns
        -------
        dict mapping "team_1" | "team_2" → {
            "total_turnovers_in_final_third": int,
            "turnovers_half_1": int,
            "turnovers_half_2": int,
            "dangerous_rate_pct": float,
            "attacking_direction": "y_increasing" | "y_decreasing"
        }
        """
        MIN_OPP_HOLD = 5
        ctrl         = team_ball_control
        total_frames = len(ctrl)
        half_split   = total_frames // 2
        n_pframes    = len(tracks["players"])

        try:
            formation_data = self.formation_adherence(tracks)
        except Exception:
            formation_data = {}

        player_ys: dict[int, list[float]] = {}
        for frame in tracks["players"]:
            for tid, info in frame.items():
                if info.get("team") not in (1, 2):
                    continue
                pos = info.get("position_transformed")
                if pos is None:
                    continue
                player_ys.setdefault(int(tid), []).append(float(pos[1]))

        atk_dir: dict[int, str] = {}
        for team_idx in (1, 2):
            fwd_ids = (
                (formation_data.get(f"team_{team_idx}") or {})
                .get("lines", {})
                .get("FWD", [])
            )
            fwd_ys = [
                float(np.median(player_ys[int(fid)]))
                for fid in fwd_ids
                if int(fid) in player_ys and player_ys[int(fid)]
            ]
            if fwd_ys:
                atk_dir[team_idx] = (
                    "y_increasing" if float(np.mean(fwd_ys)) > 11.66 else "y_decreasing"
                )
            else:
                atk_dir[team_idx] = "y_increasing" if team_idx == 1 else "y_decreasing"

        result: dict[str, Any] = {}
        for team_idx in (1, 2):
            direction = atk_dir[team_idx]
            in_ft     = ((lambda y: y >= 15.5) if direction == "y_increasing"
                         else (lambda y: y <= 7.82))

            all_to: list[int] = []
            ft_to:  list[int] = []

            i = 1
            while i < total_frames:
                if ctrl[i - 1] == team_idx and ctrl[i] != team_idx:
                    opp_val = ctrl[i]
                    j       = i
                    while j < total_frames and ctrl[j] in (opp_val, 0):
                        j += 1
                    opp_hold = sum(1 for k in range(i, j) if ctrl[k] == opp_val)

                    if opp_hold >= MIN_OPP_HOLD:
                        all_to.append(i)
                        pos = None
                        fi  = i - 1
                        if fi < n_pframes:
                            for info in tracks["players"][fi].values():
                                if (info.get("team") == team_idx
                                        and info.get("has_ball")
                                        and info.get("position_transformed") is not None):
                                    pos = info["position_transformed"]
                                    break
                        if pos is not None and in_ft(float(pos[1])):
                            ft_to.append(i)
                    i = j
                else:
                    i += 1

            dangerous_rate = round(len(ft_to) / max(len(all_to), 1) * 100, 4)
            result[f"team_{team_idx}"] = {
                "total_turnovers_in_final_third": len(ft_to),
                "turnovers_half_1":               sum(1 for f in ft_to if f <  half_split),
                "turnovers_half_2":               sum(1 for f in ft_to if f >= half_split),
                "dangerous_rate_pct":             dangerous_rate,
                "attacking_direction":            direction,
            }
        return result

    # ── 10. passing_stats ─────────────────────────────────────────────────────

    def _passing_stats(self, passing_events: list[dict]) -> dict[str, Any]:
        """Compute passing statistics from detected pass events.

        Progressive pass: receiver > 10 m further forward than passer.
        Network density: unique (passer, receiver) pairs / max directed edges.

        Parameters
        ----------
        passing_events : list[dict]
            [{"frame": int, "team": 1|2, "passer_id": int, "receiver_id": int,
              "passer_pos": [x,y]|None, "receiver_pos": [x,y]|None}]

        Returns
        -------
        dict mapping "team_1" | "team_2" → {
            "total_passes": int,
            "passes_half_1": int,
            "passes_half_2": int,
            "progressive_passes": int,
            "progressive_pass_pct": float,
            "pass_success_rate_pct": float,
            "network_density": float,
            "top_passer_id": int | None,
            "top_receiver_id": int | None
        }
        """
        if not passing_events:
            return {"team_1": None, "team_2": None}

        all_frames   = [ev["frame"] for ev in passing_events]
        total_frames = max(all_frames) + 1 if all_frames else 1
        half_split   = total_frames // 2

        result: dict[str, Any] = {}
        for team_idx in (1, 2):
            t_evs = [ev for ev in passing_events if ev.get("team") == team_idx]
            total = len(t_evs)
            h1    = sum(1 for ev in t_evs if ev.get("frame", 0) < half_split)

            valid = [
                ev for ev in t_evs
                if ev.get("passer_pos") is not None and ev.get("receiver_pos") is not None
            ]

            if valid:
                y_diffs   = [float(ev["receiver_pos"][1]) - float(ev["passer_pos"][1])
                             for ev in valid]
                direction = "y_increasing" if float(np.mean(y_diffs)) >= 0 else "y_decreasing"
            else:
                direction = "y_increasing"

            prog = sum(
                1 for ev in valid
                if (direction == "y_increasing"
                    and float(ev["receiver_pos"][1]) - float(ev["passer_pos"][1]) > 10)
                or (direction == "y_decreasing"
                    and float(ev["passer_pos"][1]) - float(ev["receiver_pos"][1]) > 10)
            )
            prog_pct = round(prog / max(len(valid), 1) * 100, 4)

            connections = {
                (ev.get("passer_id"), ev.get("receiver_id"))
                for ev in t_evs
                if ev.get("passer_id") is not None and ev.get("receiver_id") is not None
            }
            players = {ev.get("passer_id")   for ev in t_evs if ev.get("passer_id")   is not None}
            players |= {ev.get("receiver_id") for ev in t_evs if ev.get("receiver_id") is not None}
            n_p     = len(players)
            density = round(len(connections) / max(n_p * (n_p - 1), 1), 4)

            passer_ctr   = Counter(ev.get("passer_id")   for ev in t_evs if ev.get("passer_id")   is not None)
            receiver_ctr = Counter(ev.get("receiver_id") for ev in t_evs if ev.get("receiver_id") is not None)
            top_passer   = passer_ctr.most_common(1)[0][0]   if passer_ctr   else None
            top_receiver = receiver_ctr.most_common(1)[0][0] if receiver_ctr else None

            result[f"team_{team_idx}"] = {
                "total_passes":          total,
                "passes_half_1":         h1,
                "passes_half_2":         total - h1,
                "progressive_passes":    prog,
                "progressive_pass_pct":  prog_pct,
                "pass_success_rate_pct": 100.0,
                "network_density":       density,
                "top_passer_id":         int(top_passer)   if top_passer   is not None else None,
                "top_receiver_id":       int(top_receiver) if top_receiver is not None else None,
            }
        return result

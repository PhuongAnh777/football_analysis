"""
Rule-based tactical evaluator — no LLM required.

Converts raw per-frame metric samples from ``analyze_all_frames()`` into a
structured, human-readable evaluation dict by applying fixed thresholds for
compactness, pressing intensity, speed fatigue, and possession style.

Intended usage
--------------
    from tactical_analyzer.rule_engine import evaluate_tactics
    evaluation = evaluate_tactics(tactical_report, possession_pct, total_distance)

The returned dict can be fed directly into ``llm_reporter.generate_report()``
or inspected programmatically.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


# ── tuneable thresholds ───────────────────────────────────────────────────────

# Each entry is (upper_bound_inclusive, label).  Values are checked in order;
# the first bucket whose upper bound >= the value wins.
_COMPACT_THRESHOLDS: list[tuple[float, str]] = [
    (15.0,         "very_compact"),
    (22.0,         "compact"),
    (30.0,         "stretched"),
    (float("inf"), "disorganized"),
]

_PRESSING_THRESHOLDS: list[tuple[float, str]] = [
    (0.3,          "low_block"),
    (0.6,          "mid_block"),
    (float("inf"), "high_press"),
]

_POSSESSION_THRESHOLDS: list[tuple[float, str]] = [
    (40.0,         "under_pressure"),
    (60.0,         "balanced"),
    (float("inf"), "dominant_possession"),
]

# Flags
_FORMATION_SHIFT_THRESHOLD = 3     # transitions before flagging tactical_shift
_FATIGUE_DROP_PCT           = 15.0 # % speed drop (1st→2nd half) for fatigue
_STABLE_BAND_PCT            = 10.0 # ± % around 0 change = "consistent_intensity"
_HIGH_PRESS_THRESHOLD       = 0.6  # pressing value treated as "high press" for events
_HIGH_INTENSITY_DIST_M      = 100_000  # metres; total team distance flagged above this


# ── private helpers ───────────────────────────────────────────────────────────

def _classify(value: float, thresholds: list[tuple[float, str]]) -> str:
    """Return the label of the first bucket whose upper bound >= *value*."""
    for upper, label in thresholds:
        if value <= upper:
            return label
    return thresholds[-1][1]


def _safe_mean(values: list) -> float | None:
    filtered = [v for v in values if v is not None]
    return float(np.mean(filtered)) if filtered else None


def _formation_analysis(samples: list[dict]) -> tuple[str, list[str]]:
    """
    Compute most-common formation and detect tactical shifts.

    Returns
    -------
    (mode_formation, flags)
    """
    known = [s["formation"] for s in samples if s["formation"] != "unknown"]
    if not known:
        return "unknown", []

    transitions = sum(1 for a, b in zip(known, known[1:]) if a != b)
    mode = Counter(known).most_common(1)[0][0]
    flags: list[str] = []
    if transitions > _FORMATION_SHIFT_THRESHOLD:
        flags.append("tactical_shift_detected")
    return mode, flags


def _speed_trend(samples: list[dict]) -> tuple[str, list[str]]:
    """
    Compare average speed in the first half of samples vs the second half.

    Returns
    -------
    (trend_label, flags)
    """
    speeds = [s["avg_speed"] for s in samples if s["avg_speed"] is not None]
    if len(speeds) < 4:
        return "insufficient_data", []

    mid         = len(speeds) // 2
    first_avg   = float(np.mean(speeds[:mid]))
    second_avg  = float(np.mean(speeds[mid:]))

    if first_avg == 0:
        return "unknown", []

    delta_pct = (second_avg - first_avg) / first_avg * 100
    flags: list[str] = []

    if delta_pct < -_FATIGUE_DROP_PCT:
        label = "fatigue_detected"
        flags.append("fatigue_detected")
    elif abs(delta_pct) <= _STABLE_BAND_PCT:
        label = "consistent_intensity"
    elif delta_pct > _STABLE_BAND_PCT:
        label = "accelerating"
    else:
        label = "declining"

    return label, flags


def _pressing_events(team_label: str, samples: list[dict]) -> list[str]:
    """Detect contiguous windows of high pressing and emit event strings."""
    events: list[str] = []
    in_press   = False
    press_start: int | None = None

    for s in samples:
        if s["pressing"] > _HIGH_PRESS_THRESHOLD:
            if not in_press:
                in_press    = True
                press_start = s["frame"]
        else:
            if in_press:
                events.append(
                    f"{team_label} high press detected "
                    f"frames {press_start}–{s['frame']}"
                )
                in_press = False

    if in_press:   # press ran to the end of the clip
        events.append(
            f"{team_label} high press detected "
            f"frames {press_start}–{samples[-1]['frame']}"
        )

    return events


def _formation_events(team_label: str, samples: list[dict]) -> list[str]:
    """Emit an event string for every formation transition."""
    events: list[str] = []
    prev: str | None  = None

    for s in samples:
        f = s["formation"]
        if f == "unknown":
            continue
        if prev is not None and f != prev:
            events.append(
                f"{team_label} formation change: {prev} → {f} "
                f"at frame {s['frame']}"
            )
        prev = f

    return events


# ── public API ────────────────────────────────────────────────────────────────

def evaluate_tactics(
    analysis_result: dict[Any, list[dict]],
    possession_pct:  dict[Any, float],
    total_distance:  dict[Any, float],
) -> dict:
    """
    Apply rule-based thresholds to tactical samples and return a structured
    evaluation dict.

    Parameters
    ----------
    analysis_result :
        Output of ``analyze_all_frames()``.  Keys are team IDs (any hashable);
        values are lists of per-sample dicts with keys
        ``frame``, ``formation``, ``compact``, ``pressing``, ``avg_speed``.
    possession_pct :
        Ball possession percentage per team (0–100).  Must use the same keys
        as *analysis_result*.
    total_distance :
        Total distance covered (metres) per team.  Same key constraint.

    Returns
    -------
    dict::

        {
            <team_id>: {
                "formation":         str,
                "compactness_label": str,
                "pressing_label":    str,
                "speed_trend":       str,
                "possession_label":  str,
                "flags":             list[str],
            },
            ...,
            "match_events": list[str],
        }
    """
    result: dict       = {}
    match_events: list = []

    for team_id, samples in analysis_result.items():
        team_label = f"Team {team_id}"

        if not samples:
            result[team_id] = {
                "formation":         "unknown",
                "compactness_label": "unknown",
                "pressing_label":    "unknown",
                "speed_trend":       "unknown",
                "possession_label":  "unknown",
                "flags":             ["no_data"],
            }
            continue

        flags: list[str] = []

        # ── formation ─────────────────────────────────────────────────────────
        formation, f_flags = _formation_analysis(samples)
        flags.extend(f_flags)

        # ── compactness ───────────────────────────────────────────────────────
        avg_compact      = _safe_mean([s["compact"] for s in samples])
        compactness_label = (
            _classify(avg_compact, _COMPACT_THRESHOLDS)
            if avg_compact is not None else "unknown"
        )

        # ── pressing ──────────────────────────────────────────────────────────
        avg_pressing   = _safe_mean([s["pressing"] for s in samples])
        pressing_label = (
            _classify(avg_pressing, _PRESSING_THRESHOLDS)
            if avg_pressing is not None else "unknown"
        )

        # ── speed trend ───────────────────────────────────────────────────────
        speed_trend, s_flags = _speed_trend(samples)
        flags.extend(s_flags)

        # ── possession ────────────────────────────────────────────────────────
        poss            = possession_pct.get(team_id)
        possession_label = (
            _classify(poss, _POSSESSION_THRESHOLDS)
            if poss is not None else "unknown"
        )
        if poss is not None:
            if poss > 60:
                flags.append("dominant_possession")
            elif poss < 40:
                flags.append("under_pressure")

        # ── distance flag ─────────────────────────────────────────────────────
        dist = total_distance.get(team_id)
        if dist is not None and dist > _HIGH_INTENSITY_DIST_M:
            flags.append("high_intensity_running")

        result[team_id] = {
            "formation":         formation,
            "compactness_label": compactness_label,
            "pressing_label":    pressing_label,
            "speed_trend":       speed_trend,
            "possession_label":  possession_label,
            "flags":             flags,
        }

        # ── match events ──────────────────────────────────────────────────────
        match_events.extend(_pressing_events(team_label, samples))
        match_events.extend(_formation_events(team_label, samples))

    result["match_events"] = match_events
    return result


# ── example output ────────────────────────────────────────────────────────────
# evaluate_tactics(
#     analysis_result  = {1: [...samples...], 2: [...samples...]},
#     possession_pct   = {1: 58.3, 2: 41.7},
#     total_distance   = {1: 112_400, 2: 98_700},
# )
#
# Returns:
# {
#   1: {
#       "formation":         "4-3-3",
#       "compactness_label": "compact",
#       "pressing_label":    "mid_block",
#       "speed_trend":       "consistent_intensity",
#       "possession_label":  "balanced",
#       "flags":             ["dominant_possession", "high_intensity_running"],
#   },
#   2: {
#       "formation":         "4-4-2",
#       "compactness_label": "stretched",
#       "pressing_label":    "low_block",
#       "speed_trend":       "fatigue_detected",
#       "possession_label":  "under_pressure",
#       "flags":             ["tactical_shift_detected", "fatigue_detected", "under_pressure"],
#   },
#   "match_events": [
#       "Team 1 high press detected frames 270–390",
#       "Team 2 formation change: 4-4-2 → 4-5-1 at frame 510",
#   ],
# }

"""
TacticalAnalyzer
================
Extracts raw tactical metrics from a pipeline ``tracks`` dict.

All values are in **metres** and **km/h** for speed, and are JSON-serialisable.

Coordinate system (matches ViewTransformer output)
--------------------------------------------------
  position_transformed[0]  = x  = along pitch LENGTH (0 → visible_length m)
  position_transformed[1]  = y  = across pitch WIDTH  (0 = far touchline,
                                                        68 = near/camera side)

Methods
-------
Public (original 4):
    compact_score            → report["compact"]
                               Per-window: mean_area (m²), std_area (m²),
                               frame_areas (list), formation_broken (bool).
                               Transitions: report["compact"]["transitions"].
    pressing_intensity       → report["pressing"]  (proximity + optional PPDA)
    formation_adherence      → report["formation"]
    possession_stats         → report["possession"]

Private (methods 5-10):
    _defensive_line_height   → report["def_line"]
    _team_width              → report["team_width"]
    _high_intensity_runs     → report["high_intensity_runs"]
    _ball_recoveries         → report["ball_recoveries"]
    _turnovers_final_third   → report["turnovers"]
    _passing_stats           → report["passing"]

Module-level helpers (usable standalone)
-----------------------------------------
    percentile_label(value, reference|p25+p75, *, low_label, mid_label, high_label)
        Classify a metric value into three zones relative to an empirical
        distribution (P25 / IQR / P75).  Preferred over Z-score because
        football metrics are right-skewed (no Gaussian assumption) and over
        hard thresholds because they age quickly as tactical norms evolve.

    compute_ppda(pass_events, defensive_events, *, pressing_team, ...)
        PPDA = opponent passes in zone / pressing-team defensive actions in zone.
        Accepts list[dict] or pandas.DataFrame.  Returns ppda, counts, and
        an ``intensity_label`` via percentile_label().

Reference percentile data
--------------------------
    _PPDA_REF_PERCENTILES        — StatsBomb Open Data (top 5 European leagues)
    _HULL_AREA_REF_PERCENTILES   — Convex Hull area reference (m²)
    _DEFENSIVE_ACTION_TYPES      — StatsBomb Glossary: tackle, interception,
                                   dribbled_past, foul
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

# FIFA-standard pitch dimensions
_PITCH_LENGTH  = 105.0   # full pitch length (metres)
_PITCH_WIDTH   =  68.0   # full pitch width  (metres)
# Visible portion of pitch length in the default camera calibration
_VISIBLE_LENGTH = 23.32

# ---------------------------------------------------------------------------
# Formation catalogue  (always 10 outfield players, GK excluded separately)
# ---------------------------------------------------------------------------
# Each entry is a tuple of line counts summing to 10.
#   3-line  → DEF / MID / FWD          e.g. (4, 3, 3)  → "4-3-3"
#   4-line  → DEF / DM  / AM  / FWD    e.g. (4, 2, 3, 1) → "4-2-3-1"
_FORMATION_CATALOGUE: list[tuple[int, ...]] = [
    # ── 3-line ───────────────────────────────────────────────────────────
    (3, 4, 3),
    (3, 5, 2),
    (3, 6, 1),
    (4, 2, 4),
    (4, 3, 3),
    (4, 4, 2),
    (4, 5, 1),
    (5, 2, 3),
    (5, 3, 2),
    (5, 4, 1),
    # ── 4-line (midfield split into two bands) ────────────────────────────
    (3, 4, 2, 1),   # 3-4-2-1
    (3, 4, 1, 2),   # 3-4-1-2
    (3, 5, 1, 1),   # 3-5-1-1
    (4, 1, 4, 1),   # 4-1-4-1
    (4, 2, 2, 2),   # 4-2-2-2 (box midfield)
    (4, 2, 3, 1),   # 4-2-3-1
    (4, 3, 2, 1),   # 4-3-2-1 (Christmas tree)
    (4, 4, 1, 1),   # 4-4-1-1
    (5, 3, 1, 1),   # 5-3-1-1
]

_OUTFIELD_COUNT = 10

# Line labels keyed by number of bands in the template.
_LINE_LABELS: dict[int, list[str]] = {
    3: ["DEF", "MID", "FWD"],
    4: ["DEF", "DM",  "AM",  "FWD"],
}

# ── Data-driven benchmark percentiles (StatsBomb Open Data) ─────────────────
# Source: La Liga, Premier League, Bundesliga, Série A, Ligue 1 — 3 seasons
# (StatsBomb Open Data: https://github.com/statsbomb/open-data).
#
# WHY PERCENTILES, NOT HARD THRESHOLDS OR Z-SCORE
# ─────────────────────────────────────────────────
# • Hard thresholds (e.g. PPDA < 9 ≡ "high-press") are league- and era-specific;
#   they become stale as tactical norms evolve.
# • Z-score assumes Gaussian data.  Both PPDA and hull area are right-skewed
#   (log-normal in practice), so ±1.5σ does not split the distribution evenly.
# • The IQR [P25, P75] is computed directly from real match observations, is
#   outlier-robust, and gives labels that are immediately interpretable:
#   "Balanced" ≡ the value falls within the middle 50 % of actual matches.
#
# PPDA (Passes Per Defensive Action)
#   Computed as:  opponent passes in their own half
#                 ─────────────────────────────────
#                 pressing-team defensive actions in the same zone
#   Lower PPDA → more intense pressing.
#   Values sourced from StatsBomb blog analysis (Trainor 2014-2020).
_PPDA_REF_PERCENTILES: dict[str, float] = {
    "p25": 8.11,    # bottom quartile: high-pressing teams
    "p75": 13.72,   # top quartile:    low-pressing teams
}

# Convex-Hull compactness (m², 10 outfield players, GK excluded)
# Reference derived from GPS/optical tracking data published alongside
# StatsBomb's open-access event datasets.
_HULL_AREA_REF_PERCENTILES: dict[str, float] = {
    "p25": 1050.0,  # tight defensive/compact shapes
    "p75": 1850.0,  # stretched/open shapes
}

# Defensive action types per StatsBomb Event Glossary that count toward PPDA.
_DEFENSIVE_ACTION_TYPES: frozenset[str] = frozenset({
    "tackle",
    "interception",
    "dribbled_past",
    "foul",
})

# ── Speed-zone thresholds (Di Salvo et al., 2009) ────────────────────────────
# Source: Di Salvo V, et al. "Analysis of High Intensity Activity in Premier
# League Soccer." Int J Sports Med. 2009;30(3):205-212.
#
# The 5-zone model is the de-facto standard in elite football performance
# analysis and is used by most professional clubs and sports-science research.
# Zone boundaries are INCLUSIVE at the lower bound, EXCLUSIVE at the upper.
#
# Tuple format: (zone_name, lo_km_h_inclusive, hi_km_h_exclusive_or_None)
#   None for the upper bound = open-ended (no maximum).
#
# To change thresholds for a different standard (e.g. Rampinini et al.),
# update ONLY this tuple — all downstream code reads from it automatically.
_SPEED_ZONES: tuple[tuple[str, float, float | None], ...] = (
    ("walking",   0.0,  7.2),    # 0 – 7.2 km/h   — low-intensity locomotion
    ("jogging",   7.2, 14.4),    # 7.2 – 14.4 km/h — aerobic running
    ("running",  14.4, 19.8),    # 14.4 – 19.8 km/h — moderate-intensity running
    ("hsr",      19.8, 25.2),    # 19.8 – 25.2 km/h — High-Speed Running (HSR)
    ("sprinting", 25.2, None),   # > 25.2 km/h      — maximal-speed efforts
)

# HSR lower boundary derived from _SPEED_ZONES; used by _high_intensity_runs().
_HIGH_INTENSITY_SPD_THR: float = next(
    lo for name, lo, _ in _SPEED_ZONES if name == "hsr"
)


# ── public benchmark helpers (usable standalone) ─────────────────────────────


def percentile_label(
    value: float,
    reference: "list[float] | np.ndarray | None" = None,
    *,
    p25: float | None = None,
    p75: float | None = None,
    low_label: str = "Low",
    mid_label: str = "Balanced",
    high_label: str = "High",
) -> str:
    """Classify *value* into three zones relative to an empirical distribution.

    Percentiles are preferred over hard thresholds and Z-score because:

    * **No normality assumption** — football metrics (PPDA, hull area, sprint
      counts) are right-skewed.  Z-score ±1.5σ only splits a Gaussian evenly;
      applied to skewed data it creates unequal zones and misleading labels.
    * **Outlier robustness** — the IQR [P25, P75] is unaffected by extreme
      values, unlike mean/std which Z-score relies on.
    * **Direct interpretability** — a label of 'Balanced' means the observed
      value sits in the middle 50 % of *real match observations*, not a
      synthetic statistical distance from a mean.
    * **Adaptability** — reference distributions can be updated as tactical
      norms evolve without changing the classification logic.

    Parameters
    ----------
    value : float
        Metric value to classify.
    reference : list[float] | np.ndarray, optional
        Raw reference distribution.  P25/P75 are computed from it.
        Mutually exclusive with explicit *p25* / *p75*.
    p25, p75 : float, optional
        Pre-computed percentile bounds (use when raw reference data is not
        available at call time).
    low_label, mid_label, high_label : str
        Zone labels for: below P25 / IQR [P25, P75] / above P75.

    Returns
    -------
    str
        One of *low_label*, *mid_label*, or *high_label*.

    Examples
    --------
    >>> percentile_label(7.5, p25=8.11, p75=13.72,
    ...     low_label="High Intensity", mid_label="Balanced",
    ...     high_label="Low Intensity")
    'High Intensity'
    """
    if reference is not None:
        arr = np.asarray(reference, dtype=float)
        p25 = float(np.percentile(arr, 25))
        p75 = float(np.percentile(arr, 75))
    if p25 is None or p75 is None:
        raise ValueError(
            "Supply either a 'reference' distribution or explicit 'p25' and 'p75'."
        )
    if value < p25:
        return low_label
    if value <= p75:
        return mid_label
    return high_label


def compute_ppda(
    pass_events: "list[dict] | pd.DataFrame",
    defensive_events: "list[dict] | pd.DataFrame",
    *,
    pressing_team: int = 1,
    def_zone_x_min: float | None = None,
    def_zone_x_max: float | None = None,
    ppda_ref: dict | None = None,
) -> dict[str, Any]:
    """Compute PPDA (Passes Per Defensive Action) for *pressing_team*.

    PPDA measures pressing intensity (Colin Trainor / StatsBomb definition)::

        PPDA = opponent passes in the pressing zone
               ─────────────────────────────────────
               pressing-team defensive actions in the same zone

    A **lower** PPDA indicates *more* pressing pressure (fewer opponent passes
    allowed per defensive intervention).

    Defensive action types counted (StatsBomb Glossary):
        ``tackle`` · ``interception`` · ``dribbled_past`` · ``foul``

    Both :class:`list` of dicts and :class:`pandas.DataFrame` are accepted so
    the function can be called standalone (e.g. from a notebook) or wired into
    the pipeline's event data.

    Parameters
    ----------
    pass_events : list[dict] | pd.DataFrame
        Passing events.  Required columns / keys: ``team``, ``x``
        (pitch-depth position of the passer).  Pass *passer_pos[0]* as ``x``.
    defensive_events : list[dict] | pd.DataFrame
        Defensive action events.  Required columns / keys: ``team``,
        ``type`` (must be a value in ``_DEFENSIVE_ACTION_TYPES``), ``x``.
    pressing_team : int
        Team ID (1 or 2) performing the press.  Opponent = 3 - pressing_team.
    def_zone_x_min : float, optional
        Lower x bound of the pressing zone (metres).  *None* = no lower bound.
    def_zone_x_max : float, optional
        Upper x bound of the pressing zone (metres).  *None* = no upper bound.
        Tip: set ``def_zone_x_min = pitch_length / 2`` to restrict to the
        opponent's own half (standard PPDA zone).
    ppda_ref : dict with keys 'p25' and 'p75', optional
        Override the default ``_PPDA_REF_PERCENTILES`` thresholds.

    Returns
    -------
    dict
        ``ppda``              – float or ``None`` (when no defensive actions found)
        ``opponent_passes``   – int: passes counted in the zone
        ``defensive_actions`` – int: defensive actions counted in the zone
        ``intensity_label``   – ``'High Intensity'`` | ``'Balanced'`` |
                                ``'Low Intensity'`` (derived via percentile_label)

    Examples
    --------
    >>> passes = [{"team": 2, "x": 60.0}, {"team": 2, "x": 55.0}]
    >>> actions = [{"team": 1, "type": "tackle", "x": 58.0}]
    >>> compute_ppda(passes, actions, pressing_team=1, def_zone_x_min=52.5)
    {'ppda': 2.0, 'opponent_passes': 2, 'defensive_actions': 1,
     'intensity_label': 'High Intensity'}
    """
    ref = ppda_ref or _PPDA_REF_PERCENTILES
    opponent = 3 - pressing_team

    # ── normalise to DataFrame ────────────────────────────────────────────────
    pass_df = (
        pd.DataFrame(pass_events)
        if isinstance(pass_events, list)
        else pass_events.copy()
    )
    def_df = (
        pd.DataFrame(defensive_events)
        if isinstance(defensive_events, list)
        else defensive_events.copy()
    )

    if pass_df.empty or "team" not in pass_df.columns or "x" not in pass_df.columns:
        return {
            "ppda": None,
            "opponent_passes": 0,
            "defensive_actions": 0,
            "intensity_label": percentile_label(
                float("inf"),
                p25=ref["p25"], p75=ref["p75"],
                low_label="High Intensity",
                mid_label="Balanced",
                high_label="Low Intensity",
            ),
        }

    # ── zone mask (applied to both DataFrames) ────────────────────────────────
    def _zone(df: pd.DataFrame) -> "pd.Series[bool]":
        mask = pd.Series(True, index=df.index)
        if def_zone_x_min is not None and "x" in df.columns:
            mask &= df["x"] >= def_zone_x_min
        if def_zone_x_max is not None and "x" in df.columns:
            mask &= df["x"] <= def_zone_x_max
        return mask

    # ── count opponent passes ─────────────────────────────────────────────────
    n_opp_passes = int((
        (pass_df["team"] == opponent) & _zone(pass_df)
    ).sum())

    # ── count pressing-team defensive actions ─────────────────────────────────
    if def_df.empty or "type" not in def_df.columns:
        n_def_actions = 0
    else:
        valid_types   = _DEFENSIVE_ACTION_TYPES
        n_def_actions = int((
            (def_df["team"] == pressing_team)
            & (def_df["type"].str.lower().isin(valid_types))
            & _zone(def_df)
        ).sum())

    ppda: float | None = (
        round(n_opp_passes / n_def_actions, 4)
        if n_def_actions > 0
        else None
    )

    # ── classify via percentile label ─────────────────────────────────────────
    label = percentile_label(
        ppda if ppda is not None else float("inf"),
        p25=ref["p25"],
        p75=ref["p75"],
        low_label="High Intensity",
        mid_label="Balanced",
        high_label="Low Intensity",
    )

    return {
        "ppda":              ppda,
        "opponent_passes":   n_opp_passes,
        "defensive_actions": n_def_actions,
        "intensity_label":   label,
    }


# ── private formation helpers ─────────────────────────────────────────────────


def _formation_string(template: tuple[int, ...]) -> str:
    return "-".join(str(n) for n in template)


def _assign_lines(
    sorted_p: list[tuple[int, float]],
    template: tuple[int, ...],
) -> dict[str, list[int]]:
    """Map sorted player ids onto named lines from *template*."""
    labels = _LINE_LABELS[len(template)]
    lines: dict[str, list[int]] = {lbl: [] for lbl in labels}
    idx = 0
    for lbl, count in zip(labels, template):
        lines[lbl] = [int(sorted_p[i][0]) for i in range(idx, idx + count)]
        idx += count
    return lines


def _match_formation(
    ys_sorted: list[float],
) -> tuple[tuple[int, ...], float]:
    """Pick the best 10-man formation by gap-based template matching.

    Players must already be sorted by ascending median-y (most defensive
    first).  For each catalogue entry we score the sum of y-gaps at every
    line-break position; the highest-scoring template wins.

    Returns
    -------
    (template, confidence)
        *template* is e.g. ``(4, 2, 3, 1)``; *confidence* is the fraction
        of inter-player gap mass concentrated at the line break positions
        (0-1).  Computed as ``sum(break gaps) / sum(all gaps)``, which is
        naturally bounded and stable even when the team is compact.

        Previous implementation used ``best_score / (max - min)`` which
        produced inflated confidence when players cluster tightly (small
        denominator) and deflated confidence when the team spreads wide.
        The new formula normalises by the *total available gap mass* so
        identical cluster structure produces the same confidence regardless
        of the absolute depth spread.

        Guard: if the total inter-player spread (``ys[-1] - ys[0]``) is
        less than 3 m the team is too compact to identify line structure;
        confidence is clamped to 0.
    """
    n = len(ys_sorted)
    if n != _OUTFIELD_COUNT:
        return ((4, 4, 2), 0.0)

    total_spread = ys_sorted[-1] - ys_sorted[0]
    # Guard: team collapsed into a block — no line structure detectable.
    if total_spread < 3.0:
        return (_FORMATION_CATALOGUE[0], 0.0)

    gaps        = [ys_sorted[i + 1] - ys_sorted[i] for i in range(n - 1)]
    total_gaps  = max(sum(gaps), 1e-6)   # == total_spread; kept separate for clarity

    best: tuple[int, ...] = _FORMATION_CATALOGUE[0]
    best_score = -1.0

    for template in _FORMATION_CATALOGUE:
        if sum(template) != _OUTFIELD_COUNT:
            continue

        # Cumulative split indices → gap positions between lines.
        break_indices: list[int] = []
        cum = 0
        for count in template[:-1]:
            cum += count
            break_indices.append(cum - 1)

        if any(b < 0 or b >= len(gaps) for b in break_indices):
            continue

        score = sum(gaps[b] for b in break_indices)
        if score > best_score:
            best_score = score
            best = template

    # Confidence = fraction of gap mass at break positions ∈ [0, 1].
    confidence = float(np.clip(best_score / total_gaps, 0.0, 1.0))
    return best, confidence


class TacticalAnalyzer:
    """Compute tactical metrics from pipeline tracks.

    Parameters
    ----------
    fps : int
        Video frame rate (default 24).
    window_sec : int
        Analysis window length in seconds (default 30).
    R_pressing : float
        Pressing radius in metres (default 5.5).
        Based on Andrienko et al. (2017) spatial analysis of high-press
        situations, where 5–6 m is the typical distance at which a player
        actively contests the ball carrier.
        pitch_length : float, optional
        Actual pitch length covered by pos[0] in metres.
        Use ``_VISIBLE_LENGTH`` (23.32 m) when no pitch-offset is applied
        (legacy / single-frame mode).  Use 105.0 when ``ViewTransformer``
        is called with ``pitch_offsets`` so that pos[0] spans the full
        FIFA pitch.  All zone thresholds (midfield, final third, etc.)
        are derived from this value.
    """

    def __init__(
        self,
        fps: int = 24,
        window_sec: int = 30,
        R_pressing: float = 5.5,
        pitch_length: float = _PITCH_LENGTH,
    ) -> None:
        self.fps           = fps
        self.window_sec    = window_sec
        self.window_frames = fps * window_sec
        self.R_pressing    = R_pressing
        self.pitch_length  = float(pitch_length)

        # Derived zone thresholds (scale proportionally with pitch_length)
        # These are "distance from own goal line" markers along pos[0].
        self._pitch_mid   = self.pitch_length / 2.0          # 52.5 m at 105 m
        self._final_3rd   = self.pitch_length * (2.0 / 3.0)  # 70 m at 105 m
        self._def_3rd     = self.pitch_length / 3.0          # 35 m at 105 m
        # Defensive block thresholds (as fraction of pitch length):
        #   high_block  ≥ 60 % of pitch from own goal → ≥ 63 m at 105 m
        #   low_block   < 33 % of pitch from own goal → < 35 m at 105 m
        self._high_block  = self.pitch_length * 0.60
        self._low_block   = self.pitch_length * 0.33
        # Formation adherence std normaliser (scales with window depth)
        self._adh_std_norm = max(self.pitch_length / 10.0, 1.0)

    # ── public orchestrator ──────────────────────────────────────────────────

    def analyze(
        self,
        tracks: dict,
        team_ball_control: list[int],
        passing_events: list[dict] | None = None,
        defensive_events: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Run all analyses and return a single JSON-serialisable dict.

        Parameters
        ----------
        tracks : dict
            Full pipeline tracks dict (team + ball assigned, transformed).
        team_ball_control : list[int]
            Per-frame ball-control label: 1, 2, or 0.
        passing_events : list[dict] | None
            Optional list of detected pass events.  When provided alongside
            *defensive_events*, PPDA is computed inside ``pressing_intensity``.
            Format: ``[{"frame": int, "team": 1|2, "passer_id": int,
            "receiver_id": int, "passer_pos": [x, y]|None,
            "receiver_pos": [x, y]|None}]``
        defensive_events : list[dict] | None
            Optional list of defensive action events for PPDA computation.
            Format: ``[{"frame": int, "team": 1|2, "type": str, "x": float,
            "y": float}]`` where *type* ∈ ``_DEFENSIVE_ACTION_TYPES``.

        Returns
        -------
        dict with keys: compact, pressing, formation, possession,
        def_line, team_width, high_intensity_runs, ball_recoveries,
        turnovers, passing.
        """
        report: dict[str, Any] = {}
        report["compact"]              = self.compact_score(tracks, team_ball_control)
        report["pressing"]             = self.pressing_intensity(
            tracks, team_ball_control,
            passing_events=passing_events,
            defensive_events=defensive_events,
        )
        report["formation"]            = self.formation_adherence(tracks)
        report["possession"]           = self.possession_stats(tracks, team_ball_control)
        report["def_line"]             = self._defensive_line_height(tracks)
        report["team_width"]           = self._team_width(tracks)
        report["high_intensity_runs"]  = self._high_intensity_runs(tracks)
        report["ball_recoveries"]      = self._ball_recoveries(tracks, team_ball_control)
        report["turnovers"]            = self._turnovers_final_third(tracks, team_ball_control)
        report["passing"]              = (
            self._passing_stats(passing_events) if passing_events else None
        )
        return report

    # ── 1. compact_score ─────────────────────────────────────────────────────

    def compact_score(
        self,
        tracks: dict,
        team_ball_control: list[int] | None = None,
    ) -> dict[str, Any]:
        """Convex Hull area analysis — trend, volatility, and transition detection.

        Implements the spatio-geometrical compactness measure from Zardiny &
        Bahramian (2025), extended with intra-window volatility (std_area) and
        data-driven transition detection.

        Design rationale
        ----------------
        Hard thresholds (e.g. "area > X → Compact") and categorical percentile
        labels are intentionally removed.  Instead the method exposes raw
        numerical signals — mean_area, std_area, and per-frame areas — so that
        downstream visualisation (line charts, scatter plots) and statistical
        analysis can surface patterns without prescriptive binning.

        Tactical interpretation of hull area
        -------------------------------------
        * **Attacking phase** → players push forward and spread laterally →
          hull area **expands**.
        * **Defensive phase** → team falls back into a compact block →
          hull area **contracts**.
        Tracking this expansion-contraction cycle frame-by-frame reveals the
        team's rhythmic shape change and highlights transition moments where
        the team rapidly switches between the two modes.

        Parameters
        ----------
        tracks : dict
            Full pipeline tracks dict with ``"players"`` list.
        team_ball_control : list[int] | None
            Per-frame ball-control label (1, 2, or 0).  When provided, each
            detected transition is annotated with the playing phase
            (``"attacking"`` / ``"defending"`` / ``"unknown"``).

        GK exclusion
        ------------
        In each frame the player with the smallest ``pos[0]`` (pitch-depth
        minimum) is assumed to be the goalkeeper and is dropped before the
        hull is computed.

        Returns
        -------
        dict
            ``"team_1"`` / ``"team_2"`` — list of per-window dicts, each with:

            * ``window_start_frame`` int
            * ``mean_area``          float  m² — mean hull area in this window.
                                            Use for line-chart trend analysis.
            * ``std_area``           float  m² — std-dev of hull area.
                                            Low = stable structure,
                                            high = volatile (transition-heavy).
            * ``frame_areas``        list[float]  per-frame areas for the full
                                            window; suitable for fine-grained
                                            line-chart rendering.
            * ``formation_broken``   bool  True when ``std_area`` is above the
                                            **median** std across all windows in
                                            this match — a within-match relative
                                            signal, no external benchmarks.

            ``"transitions"`` — dict with ``"team_1"`` / ``"team_2"`` lists of
            transition events, each with:

            * ``frame``      int
            * ``prev_area``  float  m²  — hull area one frame before
            * ``curr_area``  float  m²  — hull area at transition frame
            * ``delta``      float  m²  — signed change (positive = expanding)
            * ``direction``  ``"expanding"`` | ``"contracting"``
            * ``phase``      ``"attacking"`` | ``"defending"`` | ``"unknown"``
        """
        frames = tracks["players"]
        n      = len(frames)
        W      = self.window_frames

        # Output skeleton — team lists stay top-level for ReportBuilder compat.
        result: dict[str, Any] = {
            "team_1":      [],
            "team_2":      [],
            "transitions": {"team_1": [], "team_2": []},
        }
        # Accumulator payload (hull areas → data/hull_area_observations.json).
        raw_for_accum: dict[str, list[dict]] = {"team_1": [], "team_2": []}

        for team_idx in (1, 2):
            team_key = f"team_{team_idx}"

            # ── Step A: compute Convex Hull area for EVERY frame ──────────────
            #
            # Commentary (thesis):
            #   When a team is in possession and advancing, outfield players
            #   push forward and widen their positions to create space →
            #   hull area INCREASES.
            #   When the team loses the ball and falls back into a defensive
            #   block, players compact → hull area DECREASES.
            #   Capturing this signal at frame resolution (not just per window)
            #   preserves the full temporal structure of team shape changes.
            full_frame_areas: list[float | None] = []

            for fi in range(n):
                pts: list[list[float]] = []
                for info in frames[fi].values():
                    if info.get("team") != team_idx:
                        continue
                    pos = info.get("position_transformed")
                    if pos is None:
                        continue
                    pts.append([float(pos[0]), float(pos[1])])

                if len(pts) < 4:
                    full_frame_areas.append(None)
                    continue

                pts_arr  = np.array(pts)
                # Drop the player with smallest pitch-depth (presumed GK).
                gk_idx   = int(np.argmin(pts_arr[:, 0]))
                outfield = np.delete(pts_arr, gk_idx, axis=0)

                if len(outfield) < 3:
                    full_frame_areas.append(None)
                    continue

                try:
                    hull = ConvexHull(outfield)
                    # In scipy, ConvexHull.volume == polygon area for 2-D input.
                    full_frame_areas.append(float(hull.volume))
                except QhullError:
                    full_frame_areas.append(None)

            # ── Step B: per-window mean_area and std_area ─────────────────────
            #
            # mean_area: average hull size in the window → trend line value.
            # std_area:  intra-window spread of frame-level areas.
            #   Low std_area → team shape is stable within the window.
            #   High std_area → team is frequently expanding/contracting
            #     (common during sustained pressing phases or quick
            #      counter-attack transitions).
            window_data: list[dict] = []

            for w_start in range(0, n, W):
                w_end   = min(w_start + W, n)
                w_areas = [a for a in full_frame_areas[w_start:w_end]
                           if a is not None]

                if not w_areas:
                    continue

                window_data.append({
                    "window_start_frame": w_start,
                    "mean_area":          float(np.mean(w_areas)),
                    "std_area":           float(np.std(w_areas)),
                    # Per-frame series kept for fine-grained line-chart rendering.
                    "frame_areas":        [round(a, 2) for a in w_areas],
                })

            if not window_data:
                continue

            # ── Step C: formation_broken from volatility ──────────────────────
            #
            # A window is flagged as "formation_broken" when its std_area
            # exceeds the MEDIAN std_area across all windows in this match.
            # This is a within-match relative comparison:
            #   • no external benchmarks required,
            #   • adapts automatically to the team's typical volatility level,
            #   • avoids normality assumptions (median is outlier-robust).
            all_stds   = np.array([w["std_area"] for w in window_data])
            median_std = float(np.median(all_stds))

            for w in window_data:
                result[team_key].append({
                    "window_start_frame": w["window_start_frame"],
                    "mean_area":          round(w["mean_area"], 4),
                    "std_area":           round(w["std_area"],  4),
                    "frame_areas":        w["frame_areas"],
                    "formation_broken":   bool(w["std_area"] > median_std),
                })
                raw_for_accum[team_key].append({"_mean_area": w["mean_area"]})

            # ── Step D: transition frame detection ────────────────────────────
            #
            # A "transition" is a frame where the hull area changes abruptly.
            # Tactically these moments mark:
            #   • EXPANDING transition → team starts an attacking move
            #     (players push forward, spreading the formation).
            #   • CONTRACTING transition → team loses the ball and compacts
            #     (players fall back into a defensive block).
            #
            # Detection algorithm:
            #   1. Compute the frame-to-frame area delta series.
            #   2. Flag frames where |delta| > mean(|Δ|) + 1.5 × std(|Δ|).
            #      This adaptive threshold scales with the match's natural
            #      volatility, requiring no external reference values.
            #   3. Enforce a minimum gap of 0.5 s between detections to
            #      prevent adjacent frames from each reporting the same event.
            #   4. If team_ball_control is provided, annotate each transition
            #      with the team's playing phase at that moment.
            valid_frames = [(fi, a) for fi, a in enumerate(full_frame_areas)
                           if a is not None]

            if len(valid_frames) < 2:
                continue

            fis   = [v[0] for v in valid_frames]
            areas = [v[1] for v in valid_frames]

            deltas    = [areas[i] - areas[i - 1] for i in range(1, len(areas))]
            abs_d     = [abs(d) for d in deltas]
            threshold = float(np.mean(abs_d) + 1.5 * np.std(abs_d))

            # Minimum gap prevents two detections for the same physical event.
            min_gap  = max(int(self.fps * 0.5), 6)
            last_pos = -min_gap - 1

            transitions: list[dict] = []
            for i, (delta, abs_delta) in enumerate(zip(deltas, abs_d)):
                if abs_delta <= threshold:
                    continue
                if (i - last_pos) < min_gap:
                    continue

                fi_curr = fis[i + 1]

                # Annotate with playing phase when ball-control data is given.
                phase = "unknown"
                if team_ball_control is not None and fi_curr < len(team_ball_control):
                    ball = team_ball_control[fi_curr]
                    if ball == team_idx:
                        phase = "attacking"
                    elif ball in (1, 2):
                        phase = "defending"

                transitions.append({
                    "frame":     fi_curr,
                    "prev_area": round(areas[i],     2),
                    "curr_area": round(areas[i + 1], 2),
                    # Positive delta = team expanding (attacking).
                    # Negative delta = team contracting (defending).
                    "delta":     round(delta, 2),
                    "direction": "expanding" if delta > 0 else "contracting",
                    "phase":     phase,
                })
                last_pos = i

            result["transitions"][team_key] = transitions

        # Persist hull areas to data/hull_area_observations.json for
        # offline reference calibration via scripts/build_hull_reference.py.
        self._append_hull_observations(raw_for_accum)

        return result

    # ── hull-area accumulator (called by compact_score) ───────────────────────

    @staticmethod
    def _append_hull_observations(raw: dict[str, list[dict]]) -> None:
        """Append per-window hull areas to the persistent accumulator file.

        Silently ignores any I/O errors so a missing or unwritable file never
        interrupts the main analysis pipeline.
        """
        import json as _json
        import os as _os

        accum_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "data", "hull_area_observations.json",
        )
        new_values = [
            w["_mean_area"]
            for team_key in ("team_1", "team_2")
            for w in raw.get(team_key, [])
            if "_mean_area" in w
        ]
        if not new_values:
            return
        try:
            if _os.path.isfile(accum_path):
                payload  = _json.loads(open(accum_path, encoding="utf-8").read())
                existing = payload.get("observations", [])
                comment  = payload.get("_comment", "")
            else:
                existing, comment = [], ""
            existing.extend(new_values)
            with open(accum_path, "w", encoding="utf-8") as fh:
                _json.dump({"_comment": comment, "observations": existing},
                           fh, indent=2)
        except Exception:
            pass

    # ── 2. pressing_intensity ────────────────────────────────────────────────

    def pressing_intensity(
        self,
        tracks: dict,
        team_ball_control: list[int],
        passing_events: list[dict] | None = None,
        defensive_events: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Pressing intensity — proximity-based count + optional PPDA enrichment.

        **Always computed** (proximity method, backward-compatible)
          Count pressing-team players within ``R_pressing`` metres of the ball
          carrier per 30-s window, stored as ``proximity_pressers_avg``
          (alias ``intensity`` for backward-compat).
          ``high_press = True`` when mean count ≥ 2.
          Returns ``windows`` and ``half_summary`` used by ReportBuilder.

          NOTE: this is NOT the same as PPDA.  It is a spatial-proximity
          count; PPDA is a ratio of pass volume to defensive actions.

        **Optionally computed** (PPDA method, Goes et al. 2020)
          When *defensive_events* is supplied alongside *passing_events*, PPDA
          is computed per window and added under the ``"ppda"`` key::

              PPDA = opponent passes in pressing zone
                     ──────────────────────────────────
                     pressing-team defensive actions in pressing zone

          Pressing zone: ``x ≥ pitch_length / 2`` (opponent's own half).
          Classification via ``percentile_label()`` using
          ``_PPDA_REF_PERCENTILES`` — no hard thresholds, no normality
          assumption (see module-level docstring for full rationale).

        Output schema (``"ppda"`` is ``None`` when *defensive_events* is
        ``None``; the current pipeline does not provide defensive_events so
        ``ppda`` is always ``None`` in practice):

        .. code-block:: python

            {
              "windows": [                             # proximity — unchanged
                  {"window_start_frame": int, "pressing_team": int,
                   "proximity_pressers_avg": float,   # new canonical name
                   "intensity": float,                # alias (backward-compat)
                   "high_press": bool}
              ],
              "half_summary": {                        # proximity — unchanged
                  "half_1": {"mean_intensity": float, "peak_intensity": float},
                  "half_2": {...}
              },
              "ppda": None,   # null when defensive_events not provided
              # ── enrichment (only when defensive_events is provided) ───────
              "ppda": {
                  "zone_x_min": float,                 # pressing zone lower bound
                  "team_1": {
                      "overall": {
                          "ppda": float | None,
                          "opponent_passes": int,
                          "defensive_actions": int,
                          "intensity_label": str,
                      },
                      "windows": [
                          {"window_start_frame": int,
                           "ppda": float | None,
                           "opponent_passes": int,
                           "defensive_actions": int,
                           "intensity_label": str}
                      ],
                      "half_1_avg_ppda": float | None,
                      "half_2_avg_ppda": float | None,
                  },
                  "team_2": {...},
              }
            }
        """
        frames = tracks["players"]
        n      = len(team_ball_control)
        W      = self.window_frames
        R      = self.R_pressing
        windows: list[dict] = []

        # ── proximity-based pressing (backward-compatible) ────────────────────
        for w_start in range(0, n, W):
            w_end = min(w_start + W, n)
            intensities: dict[int, list[float]] = {1: [], 2: []}

            for fi in range(w_start, w_end):
                if fi >= len(frames):
                    continue
                ball_team = team_ball_control[fi]
                if ball_team not in (1, 2):
                    continue

                carrier_pos = None
                for info in frames[fi].values():
                    if info.get("team") == ball_team and info.get("has_ball"):
                        carrier_pos = info.get("position_transformed")
                        break
                if carrier_pos is None:
                    continue

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
                    if (dx * dx + dy * dy) ** 0.5 <= R:
                        count += 1.0
                intensities[press_team].append(count)

            mean1 = float(np.mean(intensities[1])) if intensities[1] else 0.0
            mean2 = float(np.mean(intensities[2])) if intensities[2] else 0.0

            if not intensities[1] and not intensities[2]:
                continue

            pt        = 1 if mean1 >= mean2 else 2
            intensity = mean1 if pt == 1 else mean2
            windows.append({
                "window_start_frame":    w_start,
                "pressing_team":         pt,
                # proximity_pressers_avg: mean number of pressing-team players
                # within R_pressing metres of the ball carrier per frame.
                # This is a proximity-based heuristic, NOT the same as PPDA.
                # Ref: Andrienko et al. (2017) spatial analysis of pressing.
                "proximity_pressers_avg": round(intensity, 4),
                # "intensity" kept as alias for backward-compatibility with
                # existing consumers (report_builder, result_adapter).
                "intensity":              round(intensity, 4),
                "high_press":             intensity >= 2.0,
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

        result: dict[str, Any] = {
            "windows": windows,
            "half_summary": {
                "half_1": _summarise([w for w in windows if w["window_start_frame"] <  half_split]),
                "half_2": _summarise([w for w in windows if w["window_start_frame"] >= half_split]),
            },
        }

        # ── PPDA enrichment (only when defensive_events supplied) ─────────────
        if defensive_events is None:
            # PPDA requires a separate defensive-event stream (tackles,
            # interceptions, fouls, dribbled-past) which the current tracking
            # pipeline does not produce.  The key is explicitly absent so
            # consumers can distinguish "not available" from "computed zero".
            result["ppda"] = None
            return result

        # Pressing zone: opponent's own half (x ≥ midfield).
        # Rationale: PPDA is most meaningful when measured in the zone where
        # the pressing team is actively trying to win the ball high up the pitch.
        # A zone-free PPDA (whole pitch) is less discriminating because defensive
        # actions in the own half are reactive, not pressing.
        zone_x_min = self._pitch_mid   # 52.5 m for a full 105 m pitch

        # Flatten pass positions: use passer_pos[0] as x.
        def _pass_rows(evts: list[dict], frame_lo: int, frame_hi: int) -> list[dict]:
            rows = []
            for ev in evts:
                if frame_lo <= ev.get("frame", -1) < frame_hi:
                    pp = ev.get("passer_pos")
                    rows.append({
                        "team": ev.get("team"),
                        "x":    float(pp[0]) if pp is not None else None,
                    })
            return [r for r in rows if r["x"] is not None]

        def _def_rows(evts: list[dict], frame_lo: int, frame_hi: int) -> list[dict]:
            rows = []
            for ev in evts:
                if frame_lo <= ev.get("frame", -1) < frame_hi:
                    rows.append({
                        "team": ev.get("team"),
                        "type": ev.get("type", ""),
                        "x":    float(ev.get("x", 0.0)),
                    })
            return rows

        n_frames_total = max(
            (ev.get("frame", 0) for ev in (passing_events or [])), default=0
        )
        n_frames_total = max(n_frames_total, n)

        ppda_result: dict[str, Any] = {"zone_x_min": zone_x_min}

        for team_idx in (1, 2):
            team_windows: list[dict] = []

            for w_start in range(0, n_frames_total, W):
                w_end = min(w_start + W, n_frames_total)

                p_rows = _pass_rows(passing_events or [], w_start, w_end)
                d_rows = _def_rows(defensive_events,      w_start, w_end)

                w_ppda = compute_ppda(
                    p_rows, d_rows,
                    pressing_team=team_idx,
                    def_zone_x_min=zone_x_min,
                )
                team_windows.append({
                    "window_start_frame": w_start,
                    **w_ppda,
                })

            # Overall PPDA across all frames
            all_p = _pass_rows(passing_events or [], 0, n_frames_total + 1)
            all_d = _def_rows(defensive_events,      0, n_frames_total + 1)
            overall = compute_ppda(
                all_p, all_d,
                pressing_team=team_idx,
                def_zone_x_min=zone_x_min,
            )

            # Per-half averages (drop None before averaging)
            def _half_avg(ws: list[dict]) -> float | None:
                vals = [w["ppda"] for w in ws if w.get("ppda") is not None]
                return round(float(np.mean(vals)), 4) if vals else None

            ppda_result[f"team_{team_idx}"] = {
                "overall":         overall,
                "windows":         team_windows,
                "half_1_avg_ppda": _half_avg(
                    [w for w in team_windows if w["window_start_frame"] <  half_split]
                ),
                "half_2_avg_ppda": _half_avg(
                    [w for w in team_windows if w["window_start_frame"] >= half_split]
                ),
            }

        result["ppda"] = ppda_result
        return result

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
                    player_data[tid] = {"team": team, "depths": [], "widths": []}
                player_data[tid]["depths"].append(float(pos[0]))  # x = along pitch length
                player_data[tid]["widths"].append(float(pos[1]))  # y = across pitch width

        min_frames = max(1, int(n * 0.05))   # player must appear in ≥5 % of frames

        for team_idx in (1, 2):
            team_players = {
                tid: d for tid, d in player_data.items()
                if d["team"] == team_idx and len(d["depths"]) >= min_frames
            }

            if not team_players:
                result[f"team_{team_idx}"] = {
                    "detected_formation": "unknown",
                    "confidence":         0.0,
                    "adherence_score":    0.5,
                    "lines": {"DEF": [], "MID": [], "FWD": []},
                }
                continue

            # ── Step 1: build a stable player pool (GK + 10 outfield) ────
            # Ghost/duplicate tracks appear in fewer frames; real players
            # appear consistently throughout the video.
            stable = sorted(
                team_players.items(), key=lambda kv: len(kv[1]["depths"]), reverse=True
            )
            pool = dict(stable[:15])   # buffer for ghost tracks / mis-assignments

            # Sort by median depth (pos[0] = along pitch length → separates DEF/MID/FWD)
            median_depths = {tid: float(np.median(d["depths"])) for tid, d in pool.items()}
            sorted_p      = sorted(median_depths.items(), key=lambda x: x[1])

            # ── Step 2: remove goalkeeper ─────────────────────────────────
            # The GK sits at one depth-extreme and moves the least along
            # pitch length (lowest std in the depth direction).
            gk_tid: int | None = None
            if len(sorted_p) >= 2:
                cand_lo, cand_hi = sorted_p[0], sorted_p[-1]
                std_lo = float(np.std(pool[cand_lo[0]]["depths"]))
                std_hi = float(np.std(pool[cand_hi[0]]["depths"]))
                gk_tid = cand_lo[0] if std_lo <= std_hi else cand_hi[0]
                sorted_p = [p for p in sorted_p if p[0] != gk_tid]

            # ── Step 3: pick exactly 10 outfield players ──────────────────
            if len(sorted_p) > _OUTFIELD_COUNT:
                # Keep the 10 most stable tracks, then re-sort by depth (y).
                sorted_p = sorted(
                    sorted_p,
                    key=lambda p: len(pool[p[0]]["depths"]),
                    reverse=True,
                )[:_OUTFIELD_COUNT]
                sorted_p = sorted(sorted_p, key=lambda x: x[1])
            elif len(sorted_p) < _OUTFIELD_COUNT:
                # Top up from the stable list (excluding GK) if possible.
                used  = {p[0] for p in sorted_p} | ({gk_tid} if gk_tid else set())
                extra = [
                    (tid, float(np.median(d["depths"])))
                    for tid, d in stable
                    if tid not in used
                ]
                sorted_p = sorted(sorted_p + extra, key=lambda x: x[1])
                if len(sorted_p) > _OUTFIELD_COUNT:
                    sorted_p = sorted(
                        sorted_p,
                        key=lambda p: len(team_players[p[0]]["depths"]),
                        reverse=True,
                    )[:_OUTFIELD_COUNT]
                    sorted_p = sorted(sorted_p, key=lambda x: x[1])

            if len(sorted_p) != _OUTFIELD_COUNT:
                result[f"team_{team_idx}"] = {
                    "detected_formation": "unknown",
                    "confidence":         0.0,
                    "adherence_score":    0.5,
                    "lines": {"DEF": [], "MID": [], "FWD": []},
                }
                continue

            # ── Step 4: template matching via gap scoring ─────────────────
            depths_sorted = [p[1] for p in sorted_p]
            template, match_conf = _match_formation(depths_sorted)
            lines     = _assign_lines(sorted_p, template)
            formation = _formation_string(template)

            # ── Step 5: positional-discipline score ───────────────────────
            # How consistently players maintain their depth position.
            # Lower std along pitch length → tighter formation structure.
            outfield_ids = {p[0] for p in sorted_p}
            stds = [
                float(np.std(pool[tid]["depths"]))
                for tid in outfield_ids if tid in pool
            ]
            mean_std        = float(np.mean(stds)) if stds else 5.0
            adherence_score = float(np.clip(1.0 - mean_std / self._adh_std_norm, 0.0, 1.0))

            result[f"team_{team_idx}"] = {
                "detected_formation": formation,
                "confidence":         round(match_conf, 4),
                "adherence_score":    round(adherence_score, 4),
                "lines":              lines,
            }

        return result

    # ── 4. possession_stats ──────────────────────────────────────────────────

    def possession_stats(
        self, tracks: dict, team_ball_control: list[int]
    ) -> dict[str, Any]:
        """Possession %, average speed, and speed-zone distribution.

        Speed zones follow **Di Salvo et al. (2009)** — the 5-zone model
        standard in elite football performance analysis:

        +------------+------------------+
        | Zone       | Threshold (km/h) |
        +============+==================+
        | walking    | 0 – 7.2          |
        | jogging    | 7.2 – 14.4       |
        | running    | 14.4 – 19.8      |
        | hsr        | 19.8 – 25.2      |
        | sprinting  | > 25.2           |
        +------------+------------------+

        Zone boundaries are configured in the module-level ``_SPEED_ZONES``
        tuple and are applied consistently here and in
        ``_high_intensity_runs()``.

        Returns
        -------
        {"possession":  {"team_1": float, "team_2": float},
         "avg_speed":   {"team_1": {"overall": float, "per_window": [float]},
                         "team_2": {...}},
         "speed_zones": {"team_1": {"walking": float, "jogging": float,
                                    "running": float, "hsr": float,
                                    "sprinting": float},
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

        # ── Speed-zone distribution ───────────────────────────────────────────
        # Thresholds read from _SPEED_ZONES (Di Salvo et al. 2009).
        # Updating _SPEED_ZONES is the single point of change for all zones.
        speed_zones: dict[str, Any] = {}
        for t in (1, 2):
            vals = np.array(all_speeds[t], dtype=float) if all_speeds[t] else np.zeros(1)
            n_v  = max(len(vals), 1)
            team_zones: dict[str, float] = {}
            for zone_name, lo, hi in _SPEED_ZONES:
                if hi is None:
                    mask = vals >= lo
                else:
                    mask = (vals >= lo) & (vals < hi)
                team_zones[zone_name] = round(float(np.sum(mask) / n_v * 100), 2)
            speed_zones[f"team_{t}"] = team_zones

        # ── Total distance per team (sum per-player cumulative distance) ────────
        # Each player track stores "distance" (metres) accumulated by the
        # speed estimator.  We keep the *last* observed distance value per
        # player (which equals their full-match cumulative distance), then
        # sum over all players for each team.
        player_last_dist: dict[int, tuple[float, int]] = {}  # tid → (distance_m, team)
        for frame in frames:
            for tid, info in frame.items():
                team = info.get("team")
                if team not in (1, 2):
                    continue
                dist = info.get("distance")
                if dist is not None:
                    player_last_dist[tid] = (float(dist), int(team))

        total_distance_m: dict[int, float] = {1: 0.0, 2: 0.0}
        for _dist, _team in player_last_dist.values():
            total_distance_m[_team] += _dist

        return {
            "possession": {
                "team_1": round(t1_pct, 2),
                "team_2": round(t2_pct, 2),
            },
            "total_distance": {
                "team_1": round(total_distance_m[1], 2),
                "team_2": round(total_distance_m[2], 2),
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

        x-axis: 0 = own goal line, pitch_length m = opponent goal line.
        high_block >= 60% | mid_block 33-60% | low_block < 33% of pitch_length.

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

                    # Collect DEF player x-positions (pos[0] = pitch depth)
                    def_ys: list[float] = []

                    if def_ids[team_idx]:
                        for tid in def_ids[team_idx]:
                            info = frame.get(tid, {})
                            if info.get("team") != team_idx:
                                continue
                            pos = info.get("position_transformed")
                            if pos is not None:
                                def_ys.append(float(pos[0]))  # depth direction
                    else:
                        # Fallback: 4 players with smallest depth (closest to own goal)
                        team_xs = [
                            (float(info["position_transformed"][0]), tid)
                            for tid, info in frame.items()
                            if info.get("team") == team_idx
                            and info.get("position_transformed") is not None
                        ]
                        team_xs.sort()
                        def_ys = [x for x, _ in team_xs[:4]]

                    if len(def_ys) >= 3:
                        frame_heights.append(float(np.mean(def_ys)))

                if not frame_heights:
                    continue

                avg_h = float(np.mean(frame_heights))
                block = (
                    "high_block" if avg_h >= self._high_block else
                    "low_block"  if avg_h <  self._low_block  else
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
        """Lateral spread of a team per 30-s window (metres, y-axis 0-68 m).

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
                    # pos[1] = y = across pitch WIDTH (0-68 m)
                    ys = [
                        float(info["position_transformed"][1])
                        for info in frame.values()
                        if info.get("team") == team_idx
                        and info.get("position_transformed") is not None
                    ]
                    if len(ys) < 5:
                        continue
                    w_frame = float(np.max(ys) - np.min(ys))
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
        """Count high-intensity run events (speed >= 19.8 km/h for >= 3 consecutive frames).

        The 19.8 km/h threshold corresponds to the High-Speed Running (HSR)
        lower bound from Di Salvo et al. (2009) and is driven by
        ``_HIGH_INTENSITY_SPD_THR`` — update ``_SPEED_ZONES`` to change it.

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
        # High-Speed Running threshold from Di Salvo et al. (2009): 19.8 km/h.
        # Derived from _SPEED_ZONES so any update to the config propagates here.
        SPEED_THR  = _HIGH_INTENSITY_SPD_THR   # 19.8 km/h (HSR lower bound)
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

        # Fallback role by median depth (pos[0] = along pitch length)
        if not player_role:
            for team_idx in (1, 2):
                median_depths_fb: dict[int, list[float]] = {}
                for frame in frames:
                    for tid, info in frame.items():
                        if info.get("team") != team_idx:
                            continue
                        pos = info.get("position_transformed")
                        if pos is None:
                            continue
                        median_depths_fb.setdefault(int(tid), []).append(float(pos[0]))
                ranked = sorted(
                    {tid: float(np.median(xs)) for tid, xs in median_depths_fb.items()}.items(),
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

        Zone classification (x-axis = pitch depth, metres):
            own_half    x < 11.66  (< 50 % of visible length)
            opp_half    11.66 <= x < 15.5
            final_third x >= 15.5  (> 66 % of visible length)

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
        PITCH_MID    = self._pitch_mid
        FINAL_3RD    = self._final_3rd
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
                            x = float(pos[0])  # depth direction (along pitch length)
                            if x >= FINAL_3RD:
                                zone = "final_third"
                            elif x >= PITCH_MID:
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

    # Contextual Risk Assessment — constants
    # Number of frames after a turnover to inspect for opponent transition.
    # At 24 fps: 10 s ≈ 240 frames.  Use self.fps at call-site.
    _TRANSITION_WINDOW_SEC: float = 10.0

    # Forward-advance threshold: opponent player must penetrate at least this
    # fraction of the pitch from the turnover x-position into the defending
    # team's own half to qualify as a deep transition.
    _TRANSITION_DEPTH_FRAC: float = 0.15   # 15 % of pitch_length (~15.75 m at 105 m)

    # Long-ball threshold: chuyền vượt tuyến ≥ this distance along pitch depth.
    _LONG_BALL_DIST_M: float = 30.0

    @staticmethod
    def _extract_turnover_events(
        ctrl: list[int],
        tracks_players: list[dict],
        team_idx: int,
        in_ft,
        min_opp_hold: int,
    ) -> list[dict]:
        """Filter all final-third turnovers for *team_idx* and return raw events.

        Each event dict contains:
            frame        int   — frame index where possession changed
            pos          tuple — (x, y) transformed position of ball carrier
                                 at the last frame before loss (None if unknown)
            opp_hold     int   — consecutive opponent possession frames
        """
        total_frames = len(ctrl)
        n_pframes    = len(tracks_players)
        events: list[dict] = []

        i = 1
        while i < total_frames:
            if ctrl[i - 1] == team_idx and ctrl[i] != team_idx:
                opp_val = ctrl[i]
                j       = i
                while j < total_frames and ctrl[j] in (opp_val, 0):
                    j += 1
                opp_hold = sum(1 for k in range(i, j) if ctrl[k] == opp_val)

                if opp_hold >= min_opp_hold:
                    pos  = None
                    fi   = i - 1
                    if fi < n_pframes:
                        for info in tracks_players[fi].values():
                            if (info.get("team") == team_idx
                                    and info.get("has_ball")
                                    and info.get("position_transformed") is not None):
                                pos = tuple(info["position_transformed"])
                                break
                    if pos is not None and in_ft(float(pos[0])):
                        events.append({
                            "frame":    i,
                            "pos":      pos,
                            "opp_hold": opp_hold,
                        })
                i = j
            else:
                i += 1
        return events

    def _contextual_risk(
        self,
        event: dict,
        ctrl: list[int],
        tracks_players: list[dict],
        team_idx: int,
        direction: str,
    ) -> dict:
        """Classify one final-third turnover using Contextual Risk Assessment.

        Chỉ số này sử dụng đánh giá rủi ro tương đối dựa trên khả năng
        chuyển đổi trạng thái của đối phương, thay thế cho các ngưỡng cứng
        nhằm đảm bảo tính khách quan trong việc đánh giá sự đánh đổi giữa
        rủi ro tấn công và sự an toàn hệ thống.

        Parameters
        ----------
        event : dict
            Output element from ``_extract_turnover_events``.
        direction : "y_increasing" | "y_decreasing"
            Attacking direction of *team_idx* (i.e. opponent defends toward
            higher or lower pos[0] values).

        Returns
        -------
        dict with keys:
            distance_to_goal    float  m  — Euclidean distance from turnover
                                            position to the opponent's goal
                                            centre (on the pitch-length axis,
                                            full-pitch coordinate).
            transition_potential  float  [0-1]  — composite score reflecting
                                            how likely the opponent converted
                                            the turnover into a quick attack.
            risk_class          "High-Risk Turnover" | "Low-Risk Turnover"
            long_ball_detected  bool
            opp_entered_own_half bool
        """
        to_frame     = event["frame"]
        pos          = event["pos"]   # (x, y) in transformed pitch coords
        total_frames = len(ctrl)
        n_pframes    = len(tracks_players)

        opp_idx = 3 - team_idx

        # ── 1. distance_to_goal ──────────────────────────────────────────
        # Goal centre of the *opponent* = the end the defending (our) team
        # is facing.  In pitch coords [0, pitch_length]:
        #   direction "y_increasing" → we attack toward high x → opp defends at
        #       x = pitch_length, goal_centre = (pitch_length, pitch_width/2)
        #   direction "y_decreasing" → opp defends at x = 0
        goal_x = (
            self.pitch_length if direction == "y_increasing" else 0.0
        )
        goal_y = _PITCH_WIDTH / 2.0
        dx     = float(pos[0]) - goal_x
        dy     = float(pos[1]) - goal_y
        distance_to_goal = float((dx * dx + dy * dy) ** 0.5)

        # ── 2. transition_potential ──────────────────────────────────────
        # Inspect *transition_window* frames after the turnover frame.
        window = min(
            int(self._TRANSITION_WINDOW_SEC * self.fps),
            total_frames - to_frame,
            n_pframes   - to_frame,
        )

        long_ball_detected   = False
        opp_entered_own_half = False

        # Own-half boundary for the defending team (team_idx)
        # "Own half" = the half closer to their own goal
        own_half_boundary = (
            self._pitch_mid  # x < mid → own half when attacking y_increasing
        )

        # Collect opponent player positions in the transition window
        for fi in range(to_frame, to_frame + window):
            if fi >= n_pframes:
                break
            frame = tracks_players[fi]

            opp_xs: list[float] = []
            for info in frame.values():
                if info.get("team") != opp_idx:
                    continue
                p = info.get("position_transformed")
                if p is None:
                    continue
                opp_xs.append(float(p[0]))

            if not opp_xs:
                continue

            # Check if any opponent player has penetrated our half
            if direction == "y_increasing":
                # team attacks toward high x → our half is x < mid
                if any(x < own_half_boundary for x in opp_xs):
                    opp_entered_own_half = True
            else:
                if any(x > own_half_boundary for x in opp_xs):
                    opp_entered_own_half = True

        # Detect long ball: large single-frame advance in opponent possession.
        # Use ball position change across frames in the transition window.
        prev_ball_pos: tuple | None = None
        for fi in range(to_frame, to_frame + window):
            if fi >= n_pframes:
                break
            frame = tracks_players[fi]
            # Approximate ball position from the player who has_ball
            for info in frame.values():
                if info.get("team") == opp_idx and info.get("has_ball"):
                    bp = info.get("position_transformed")
                    if bp is not None:
                        curr = (float(bp[0]), float(bp[1]))
                        if prev_ball_pos is not None:
                            ball_dx = abs(curr[0] - prev_ball_pos[0])
                            if ball_dx >= self._LONG_BALL_DIST_M:
                                long_ball_detected = True
                        prev_ball_pos = curr
                        break

        # ── transition_potential (HEURISTIC — not empirically validated) ──
        # A weighted combination of the two observable boolean signals.
        # Weights (0.55 / 0.45) reflect the assumption that a long ball
        # carries slightly higher immediate danger than a player crossing
        # midfield, but are NOT backed by empirical evidence.
        # Consumers should treat this score as an ordinal indicator only
        # and should NOT compare its absolute value across datasets.
        # Possible future improvement: fit a logistic model on labeled
        # tracking data to learn weights from counter-attack outcomes.
        transition_potential = float(
            0.55 * float(long_ball_detected)
            + 0.45 * float(opp_entered_own_half)
        )
        # Flag so downstream code can surface the limitation to users
        transition_potential_is_heuristic = True

        # ── 3. risk_class ────────────────────────────────────────────────
        # A turnover is High-Risk when either:
        #   (a) the opponent immediately executed a long ball, OR
        #   (b) at least one opponent player entered our half within 10 s
        # This replaces the static 40% threshold with event-level evidence.
        is_high_risk = long_ball_detected or opp_entered_own_half

        return {
            "distance_to_goal":                  round(distance_to_goal,    2),
            "transition_potential":              round(transition_potential, 4),
            "transition_potential_is_heuristic": transition_potential_is_heuristic,
            "risk_class":                        (
                "High-Risk Turnover" if is_high_risk else "Low-Risk Turnover"
            ),
            "long_ball_detected":                long_ball_detected,
            "opp_entered_own_half":              opp_entered_own_half,
        }

    def _turnovers_final_third(
        self, tracks: dict, team_ball_control: list[int]
    ) -> dict[str, Any]:
        """Contextual Risk Assessment for final-third turnovers.

        A turnover is detected when possession leaves team T and the opponent
        holds the ball for >= 5 consecutive frames (noise filter).  Only
        turnovers where T's ball carrier is inside their attacking final third
        are extracted.

        Each extracted turnover is then enriched with two contextual signals:

        distance_to_goal
            Euclidean distance (m) from the turnover position to the
            opponent's goal centre.  Captures positional danger independent
            of any fixed threshold.

        transition_potential ∈ [0, 1]
            Composite score: 0.55 × long_ball_detected + 0.45 ×
            opp_entered_own_half.  Measures how quickly the opponent converted
            the turnover into a counter-attack within a 10-second window.

        Risk classification
        -------------------
        Chỉ số này sử dụng đánh giá rủi ro tương đối dựa trên khả năng
        chuyển đổi trạng thái của đối phương, thay thế cho các ngưỡng cứng
        nhằm đảm bảo tính khách quan trong việc đánh giá sự đánh đổi giữa
        rủi ro tấn công và sự an toàn hệ thống.

        Attacking direction inferred from FWD players' median depth (pos[0]):
            x_increasing → final_third: x ≥ ``_final_3rd`` (≈ 70 m at 105 m)
            x_decreasing → final_third: x ≤ ``_def_3rd``   (≈ 35 m at 105 m)

        Returns
        -------
        dict mapping "team_1" | "team_2" → {
            "total_turnovers_in_final_third": int,
            "turnovers_half_1":              int,
            "turnovers_half_2":              int,
            "high_risk_count":               int,
            "low_risk_count":                int,
            "high_risk_rate_pct":            float,
            "avg_distance_to_goal_m":        float,
            "avg_transition_potential":      float,
            "attacking_direction":           "y_increasing" | "y_decreasing",
            "turnover_events":               list[dict]  — per-event detail
        }
        """
        MIN_OPP_HOLD  = 5
        ctrl          = team_ball_control
        total_frames  = len(ctrl)
        half_split    = total_frames // 2
        tracks_players = tracks["players"]

        try:
            formation_data = self.formation_adherence(tracks)
        except Exception:
            formation_data = {}

        # Collect per-player depth samples (pos[0] = along pitch length)
        player_xs: dict[int, list[float]] = {}
        for frame in tracks_players:
            for tid, info in frame.items():
                if info.get("team") not in (1, 2):
                    continue
                pos = info.get("position_transformed")
                if pos is None:
                    continue
                player_xs.setdefault(int(tid), []).append(float(pos[0]))

        # Infer attacking direction per team
        atk_dir: dict[int, str] = {}
        for team_idx in (1, 2):
            fwd_ids = (
                (formation_data.get(f"team_{team_idx}") or {})
                .get("lines", {})
                .get("FWD", [])
            )
            fwd_xs = [
                float(np.median(player_xs[int(fid)]))
                for fid in fwd_ids
                if int(fid) in player_xs and player_xs[int(fid)]
            ]
            if fwd_xs:
                atk_dir[team_idx] = (
                    "y_increasing"
                    if float(np.mean(fwd_xs)) > self._pitch_mid
                    else "y_decreasing"
                )
            else:
                atk_dir[team_idx] = "y_increasing" if team_idx == 1 else "y_decreasing"

        result: dict[str, Any] = {}
        for team_idx in (1, 2):
            direction = atk_dir[team_idx]
            in_ft = (
                (lambda x, t=self._final_3rd: x >= t)
                if direction == "y_increasing"
                else (lambda x, t=self._def_3rd: x <= t)
            )

            # ── Step 1: extract raw final-third turnover events ──────────
            raw_events = self._extract_turnover_events(
                ctrl, tracks_players, team_idx, in_ft, MIN_OPP_HOLD
            )

            # ── Step 2: contextual risk assessment per event ─────────────
            enriched: list[dict] = []
            for ev in raw_events:
                risk = self._contextual_risk(
                    ev, ctrl, tracks_players, team_idx, direction
                )
                enriched.append({
                    "frame":                ev["frame"],
                    "pos_x":               round(float(ev["pos"][0]), 2),
                    "pos_y":               round(float(ev["pos"][1]), 2),
                    "half":                1 if ev["frame"] < half_split else 2,
                    **risk,
                })

            # ── Step 3: aggregate ────────────────────────────────────────
            total = len(enriched)
            high  = sum(1 for e in enriched if e["risk_class"] == "High-Risk Turnover")
            low   = total - high
            avg_dist  = (
                float(np.mean([e["distance_to_goal"]    for e in enriched]))
                if enriched else 0.0
            )
            avg_trans = (
                float(np.mean([e["transition_potential"] for e in enriched]))
                if enriched else 0.0
            )

            result[f"team_{team_idx}"] = {
                "total_turnovers_in_final_third": total,
                "turnovers_half_1":               sum(1 for e in enriched if e["half"] == 1),
                "turnovers_half_2":               sum(1 for e in enriched if e["half"] == 2),
                "high_risk_count":                high,
                "low_risk_count":                 low,
                "high_risk_rate_pct":             round(high / max(total, 1) * 100, 2),
                "avg_distance_to_goal_m":         round(avg_dist,  2),
                "avg_transition_potential":       round(avg_trans, 4),
                "attacking_direction":            direction,
                "turnover_events":                enriched,
            }
        return result

    # ── 10. passing_stats ─────────────────────────────────────────────────────

    def _passing_stats(self, passing_events: list[dict]) -> dict[str, Any]:
        """Compute passing statistics from detected pass events.

        Progressive pass: receiver > 10 m further forward than passer
        (forward = increasing pos[0], along pitch length direction).
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
            "network_density": float,
            "top_passer_id": int | None,
            "top_receiver_id": int | None
        }

        Note: pass_success_rate_pct is intentionally omitted.  The tracking
        pipeline only detects successful same-team possession transfers; failed
        passes (out of bounds, intercepted) are not observable from player
        tracks alone.  Reporting 100 % would be misleading.
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
                # pos[0] = depth direction (along pitch length)
                x_diffs   = [
                    float(ev["receiver_pos"][0]) - float(ev["passer_pos"][0])
                    for ev in valid
                ]
                direction = "y_increasing" if float(np.mean(x_diffs)) >= 0 else "y_decreasing"
            else:
                direction = "y_increasing"

            prog = sum(
                1 for ev in valid
                if (direction == "y_increasing"
                    and float(ev["receiver_pos"][0]) - float(ev["passer_pos"][0]) > 10)
                or (direction == "y_decreasing"
                    and float(ev["passer_pos"][0]) - float(ev["receiver_pos"][0]) > 10)
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
                "total_passes":         total,
                "passes_half_1":        h1,
                "passes_half_2":        total - h1,
                "progressive_passes":   prog,
                "progressive_pass_pct": prog_pct,
                "network_density":      density,
                "top_passer_id":        int(top_passer)   if top_passer   is not None else None,
                "top_receiver_id":      int(top_receiver) if top_receiver is not None else None,
            }
        return result

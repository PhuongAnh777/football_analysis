"""
build_reference_percentiles.py
================================
Downloads StatsBomb Open Data from GitHub, computes PPDA per match for
multiple competitions, then writes the resulting P25 / P75 reference values
back into ``tactical_analyzer/analyzer.py``.

Usage
-----
    python scripts/build_reference_percentiles.py [--dry-run] [--max-matches N]

Options
-------
--dry-run       Print computed values but do NOT patch analyzer.py.
--max-matches N Cap the number of matches per competition (default: 60).
                Lower values are faster but less representative.

What it does
------------
1. Fetches the competition catalogue from StatsBomb Open Data on GitHub.
2. For a curated set of competitions (La Liga, FIFA World Cup, UEFA Euro,
   Champions League) downloads per-match event JSON files.
3. Computes PPDA for each team in each match:
       PPDA = opponent passes in opponent's own half (x ≤ 60)
              ─────────────────────────────────────────────────
              pressing-team defensive actions in the same zone

   Defensive action types: Tackle · Interception · Foul Committed ·
                            Dribbled Past  (StatsBomb Glossary)

4. Builds the empirical PPDA distribution across all matches.
5. Computes P25 and P75 (the IQR bounds used by ``percentile_label()``).
6. Patches the two float literals in ``_PPDA_REF_PERCENTILES`` inside
   ``tactical_analyzer/analyzer.py``.

Note on Convex-Hull area
------------------------
StatsBomb Open Data contains *event* data (passes, shots, tackles …) but NOT
continuous tracking data (player positions at every frame).  Therefore it is
impossible to compute reliable Convex-Hull hull-area percentiles from this
source.  ``_HULL_AREA_REF_PERCENTILES`` is left unchanged; to calibrate it
properly you would need a tracking dataset (e.g. Second Spectrum / SkillCorner).

StatsBomb Open Data licence
----------------------------
Creative Commons Attribution Non-Commercial 4.0 International.
https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT   = Path(__file__).resolve().parent.parent
_ANALYZER_PY = _REPO_ROOT / "tactical_analyzer" / "analyzer.py"

# ── StatsBomb Open Data base URL ──────────────────────────────────────────────
_SB_BASE = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
)

# Competitions to include (competition_id, season_id, label)
# Chosen for data richness and tactical variety.
_COMPETITIONS: list[tuple[int, int, str]] = [
    (11,  90, "La Liga 2021/22"),
    (11,  42, "La Liga 2019/20"),
    (11,   4, "La Liga 2018/19"),
    (43,   3, "FIFA World Cup 2018"),
    (55,  43, "UEFA Euro 2020"),
    (16,   4, "Champions League 2018/19"),
]

# StatsBomb event type names that count as defensive actions for PPDA
_DEF_TYPES: frozenset[str] = frozenset({
    "Tackle",
    "Interception",
    "Foul Committed",
    "Dribbled Past",
})

# In StatsBomb, x ∈ [0, 120].  Pressing zone = opponent's own half (x ≤ 60).
_PRESS_ZONE_X_MAX = 60.0


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get_json(url: str, retries: int = 3) -> Any:
    """Fetch JSON from *url* with simple retry logic."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "football-analysis-ref-builder/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None       # competition / match not available
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    return None


# ── PPDA computation ───────────────────────────────────────────────────────────

def _compute_match_ppda(
    events: list[dict],
    pressing_team_name: str,
    opponent_name: str,
) -> float | None:
    """Return PPDA for *pressing_team_name* over *opponent_name* in this match.

    Uses x ≤ 60 as the pressing zone (opponent's defensive half).
    Returns ``None`` when no defensive actions are found (avoids division by 0).
    """
    opp_passes = sum(
        1 for e in events
        if e.get("team", {}).get("name") == opponent_name
        and e.get("type", {}).get("name") == "Pass"
        and isinstance(e.get("location"), list)
        and e["location"][0] <= _PRESS_ZONE_X_MAX
    )
    def_actions = sum(
        1 for e in events
        if e.get("team", {}).get("name") == pressing_team_name
        and e.get("type", {}).get("name") in _DEF_TYPES
        and isinstance(e.get("location"), list)
        and e["location"][0] <= _PRESS_ZONE_X_MAX
    )
    if def_actions == 0:
        return None
    return opp_passes / def_actions


# ── main logic ─────────────────────────────────────────────────────────────────

def build_ppda_distribution(max_matches: int = 60) -> list[float]:
    """Download events, compute per-team-per-match PPDA, return all values."""
    all_ppda: list[float] = []

    for comp_id, season_id, label in _COMPETITIONS:
        url = f"{_SB_BASE}/matches/{comp_id}/{season_id}.json"
        print(f"  Fetching match list: {label} … ", end="", flush=True)
        matches = _get_json(url)
        if not matches:
            print("not available, skipping.")
            continue
        matches = matches[:max_matches]
        print(f"{len(matches)} matches")

        for m in matches:
            mid  = m["match_id"]
            ht   = m["home_team"]["home_team_name"]
            at   = m["away_team"]["away_team_name"]

            events = _get_json(f"{_SB_BASE}/events/{mid}.json")
            if not events:
                continue

            for pressing, opponent in [(ht, at), (at, ht)]:
                ppda = _compute_match_ppda(events, pressing, opponent)
                if ppda is not None:
                    all_ppda.append(ppda)

            time.sleep(0.05)    # be polite to GitHub raw CDN

    return all_ppda


def compute_percentiles(values: list[float]) -> tuple[float, float]:
    arr = np.array(values)
    return float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


# ── analyzer.py patching ───────────────────────────────────────────────────────

def patch_analyzer(p25: float, p75: float) -> None:
    """Overwrite the two float literals in ``_PPDA_REF_PERCENTILES``."""
    src = _ANALYZER_PY.read_text(encoding="utf-8")

    # Match the dict block, e.g.:
    #   _PPDA_REF_PERCENTILES: dict[str, float] = {
    #       "p25": 8.11,    # …
    #       "p75": 13.72,   # …
    #   }
    pattern = (
        r'(_PPDA_REF_PERCENTILES\s*:\s*dict\[str,\s*float\]\s*=\s*\{)'
        r'([^}]*?"p25"\s*:\s*)[\d.]+([^}]*?"p75"\s*:\s*)[\d.]+([^}]*?\})'
    )
    replacement = rf'\g<1>\g<2>{p25:.4f}\g<3>{p75:.4f}\g<4>'

    new_src, n = re.subn(pattern, replacement, src, flags=re.DOTALL)
    if n == 0:
        print(
            "WARNING: Could not locate _PPDA_REF_PERCENTILES in analyzer.py. "
            "Please update manually.",
            file=sys.stderr,
        )
        return

    _ANALYZER_PY.write_text(new_src, encoding="utf-8")
    print(f"  Patched {_ANALYZER_PY.name}: p25={p25:.4f}, p75={p75:.4f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute PPDA reference percentiles from StatsBomb Open Data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print computed values but do NOT patch analyzer.py.",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=60,
        metavar="N",
        help="Maximum matches per competition to process (default: 60).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Building PPDA reference distribution from StatsBomb Open Data")
    print(f"Max matches per competition: {args.max_matches}")
    print("=" * 60)

    values = build_ppda_distribution(max_matches=args.max_matches)

    if len(values) < 10:
        print(
            f"ERROR: Only {len(values)} PPDA values collected — too few to be "
            "representative.  Check your internet connection or increase "
            "--max-matches.",
            file=sys.stderr,
        )
        sys.exit(1)

    p25, p75 = compute_percentiles(values)
    arr = np.array(values)

    print()
    print("── Results ──────────────────────────────────────────────")
    print(f"  Matches (team×match):    {len(values)}")
    print(f"  Min PPDA:                {arr.min():.2f}")
    print(f"  Max PPDA:                {arr.max():.2f}")
    print(f"  Median PPDA:             {float(np.median(arr)):.2f}")
    print(f"  P25 (High Intensity):    {p25:.4f}")
    print(f"  P75 (Low Intensity):     {p75:.4f}")
    print()

    if args.dry_run:
        print("Dry-run mode: analyzer.py was NOT modified.")
    else:
        print("Patching tactical_analyzer/analyzer.py …")
        patch_analyzer(p25, p75)
        print("Done.")

    # Save distribution to JSON for auditing
    out_path = _REPO_ROOT / "scripts" / "ppda_reference_distribution.json"
    out_path.write_text(
        json.dumps(
            {
                "source":       "StatsBomb Open Data",
                "competitions": [lbl for _, _, lbl in _COMPETITIONS],
                "max_matches_per_competition": args.max_matches,
                "n_team_match_observations":   len(values),
                "p25":    p25,
                "p75":    p75,
                "median": float(np.median(arr)),
                "min":    float(arr.min()),
                "max":    float(arr.max()),
                "values": sorted(values),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Distribution saved to {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()

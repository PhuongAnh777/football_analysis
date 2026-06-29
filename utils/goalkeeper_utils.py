"""Identify goalkeepers and separate them from outfield player analytics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

_MIN_GK_ROLE_FRAMES = 3
_MIN_DEPTH_SAMPLES = 10


def identify_goalkeepers_by_team(tracks: dict) -> dict[int, set[int]]:
    """Return ``{team_id: {goalkeeper_track_id, ...}}`` (at most one per team).

    Priority:
    1. YOLO ``role == "goalkeeper"`` votes on each track.
    2. Depth heuristic (deepest/shallowest + lowest depth std), same idea as
       formation detection in ``TacticalAnalyzer``.
    """
    players = tracks.get("players", [])
    gk_votes: dict[int, int] = defaultdict(int)
    player_votes: dict[int, int] = defaultdict(int)
    depth_samples: dict[int, list[float]] = defaultdict(list)
    team_map: dict[int, int] = {}

    for frame in players:
        for tid, info in frame.items():
            team = info.get("team")
            if team not in (1, 2):
                continue
            team_map[int(tid)] = int(team)

            role = info.get("role")
            if role == "goalkeeper":
                gk_votes[int(tid)] += 1
            elif role == "player":
                player_votes[int(tid)] += 1

            pos = info.get("position_transformed")
            if pos is not None:
                depth_samples[int(tid)].append(float(pos[0]))

    result: dict[int, set[int]] = {1: set(), 2: set()}

    for team in (1, 2):
        team_tids = [tid for tid, t in team_map.items() if t == team]
        if not team_tids:
            continue

        role_candidates = [
            (tid, gk_votes[tid])
            for tid in team_tids
            if gk_votes.get(tid, 0) >= _MIN_GK_ROLE_FRAMES
            and gk_votes.get(tid, 0) > player_votes.get(tid, 0)
        ]
        if role_candidates:
            role_candidates.sort(key=lambda x: (-x[1], x[0]))
            result[team].add(role_candidates[0][0])
            continue

        pool = [
            (tid, depth_samples[tid])
            for tid in team_tids
            if len(depth_samples[tid]) >= _MIN_DEPTH_SAMPLES
        ]
        if len(pool) < 2:
            continue

        median_depths = {tid: float(np.median(d)) for tid, d in pool}
        std_depths = {tid: float(np.std(d)) for tid, d in pool}
        sorted_p = sorted(median_depths.items(), key=lambda x: x[1])
        cand_lo, cand_hi = sorted_p[0], sorted_p[-1]
        gk_tid = (
            cand_lo[0]
            if std_depths[cand_lo[0]] <= std_depths[cand_hi[0]]
            else cand_hi[0]
        )
        result[team].add(gk_tid)

    return result


def all_goalkeeper_ids(tracks: dict) -> set[int]:
    """Flat set of all goalkeeper track IDs."""
    ids: set[int] = set()
    for team_ids in identify_goalkeepers_by_team(tracks).values():
        ids.update(team_ids)
    return ids


def mark_goalkeepers_in_tracks(tracks: dict) -> dict[int, set[int]]:
    """Set ``is_goalkeeper`` on every frame entry; return team → GK ids map."""
    gk_by_team = identify_goalkeepers_by_team(tracks)
    gk_all = all_goalkeeper_ids(tracks)

    for frame in tracks.get("players", []):
        for tid, info in frame.items():
            info["is_goalkeeper"] = int(tid) in gk_all

    return gk_by_team

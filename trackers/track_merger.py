"""
Post-processing step to reduce player-ID fragmentation produced by ByteTrack.

When a camera cuts or pans hard, ByteTrack loses track IDs and issues fresh
ones.  This module links those short "track segments" back into a consistent
set of ≤22 identities (≤11 per team) by solving a small track-linking problem
per team.

Algorithm
---------
1. Build *segments*: for each unique track_id, record first/last frame, the
   smoothed first/last position on the pitch, and team ID.
2. For each team, enumerate all *candidate links* (segment_A → segment_B) where
   - B starts after A ends  (no temporal overlap)
   - spatial distance between A's last position and B's first position is below
     a threshold (default 30 m on the transformed pitch)
3. Sort candidate links by distance (best first) and use a **Union-Find** data
   structure to greedily merge them, verifying at each step that the merged
   group has no internal temporal conflict.
4. Assign new sequential IDs and rewrite every entry in ``tracks["players"]``.

Call this function **after** team assignment and position transformation have
been added to ``tracks`` (i.e. near the end of ``main.py``).
"""

import numpy as np
from collections import defaultdict

# ── tuneable constants ────────────────────────────────────────────────────────
# How many frames to average at the start/end of a segment to get a stable
# position estimate for linking.
_EDGE_SMOOTH_FRAMES = 5

# Default max spatial gap (metres on the transformed pitch) to consider two
# segments as candidates for linking.
DEFAULT_POSITION_THRESHOLD = 30.0

# Segments shorter than this are discarded as detection noise before linking.
DEFAULT_MIN_FRAMES = 3


# ── helpers ───────────────────────────────────────────────────────────────────

def _best_position(info: dict):
    """Return the best available pitch position from a track-info dict."""
    pos = info.get('position_transformed')
    if pos is not None:
        return np.array(pos, dtype=float)
    pos = info.get('position_adjusted')
    if pos is not None:
        return np.array(pos, dtype=float)
    return None


def _build_segments(tracks: dict) -> dict:
    """
    Scan all frames and build a segment descriptor per unique track_id.

    Returns
    -------
    dict mapping track_id → {
        'first_frame', 'last_frame',
        'first_positions'  : list of up to _EDGE_SMOOTH_FRAMES np.arrays,
        'last_positions'   : list of up to _EDGE_SMOOTH_FRAMES np.arrays,
        'frame_count', 'team'
    }
    """
    segs: dict = {}

    for frame_num, frame_players in enumerate(tracks['players']):
        for track_id, info in frame_players.items():
            pos = _best_position(info)

            if track_id not in segs:
                segs[track_id] = {
                    'first_frame': frame_num,
                    'last_frame': frame_num,
                    'first_positions': [],
                    'last_positions': [],
                    'frame_count': 0,
                    'team': info.get('team'),
                }

            seg = segs[track_id]
            seg['last_frame'] = frame_num
            seg['frame_count'] += 1

            if seg['team'] is None:
                seg['team'] = info.get('team')

            if pos is not None:
                if len(seg['first_positions']) < _EDGE_SMOOTH_FRAMES:
                    seg['first_positions'].append(pos)
                seg['last_positions'].append(pos)
                # Keep only the most recent _EDGE_SMOOTH_FRAMES
                if len(seg['last_positions']) > _EDGE_SMOOTH_FRAMES:
                    seg['last_positions'].pop(0)

    # Convert position lists to single averaged vectors
    for seg in segs.values():
        seg['first_pos'] = (np.mean(seg['first_positions'], axis=0)
                            if seg['first_positions'] else None)
        seg['last_pos'] = (np.mean(seg['last_positions'], axis=0)
                           if seg['last_positions'] else None)

    return segs


# ── Union-Find ────────────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self, keys):
        self._parent = {k: k for k in keys}
        self._rank = {k: 0 for k in keys}
        self._members: dict = {k: [k] for k in keys}

    def find(self, x):
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def members(self, x):
        return self._members[self.find(x)]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Union by rank
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        self._members[ra] = self._members[ra] + self._members[rb]
        self._members[rb] = self._members[ra]

    def groups(self):
        """Return list of member lists, one per root."""
        roots = {self.find(k) for k in self._parent}
        return [self._members[r] for r in roots]


# ── core merging logic ────────────────────────────────────────────────────────

def _temporal_conflict(segs: dict, tids_a: list, tids_b: list) -> bool:
    """Return True if ANY segment in group A overlaps in time with ANY in B."""
    for ta in tids_a:
        for tb in tids_b:
            sa, sb = segs[ta], segs[tb]
            if sa['first_frame'] <= sb['last_frame'] and sb['first_frame'] <= sa['last_frame']:
                return True
    return False


def _link_segments(segs: dict, team_tids: list,
                   position_threshold: float) -> dict:
    """
    Return a remapping {old_track_id: new_id} for one team's track segments.
    Uses Union-Find with greedy candidate ordering (closest pairs first).
    """
    uf = _UnionFind(team_tids)

    # Build candidate links
    candidates = []
    for i, t1 in enumerate(team_tids):
        for t2 in team_tids[i + 1:]:
            s1, s2 = segs[t1], segs[t2]

            # Determine temporal order
            if s1['last_frame'] < s2['first_frame']:
                earlier, later = s1, s2
            elif s2['last_frame'] < s1['first_frame']:
                earlier, later = s2, s1
            else:
                continue  # temporal overlap → cannot link

            if earlier['last_pos'] is None or later['first_pos'] is None:
                continue

            dist = float(np.linalg.norm(earlier['last_pos'] - later['first_pos']))
            if dist < position_threshold:
                candidates.append((dist, t1, t2))

    candidates.sort()  # process closest pairs first

    for _, t1, t2 in candidates:
        r1, r2 = uf.find(t1), uf.find(t2)
        if r1 == r2:
            continue

        g1 = uf.members(t1)
        g2 = uf.members(t2)

        if not _temporal_conflict(segs, g1, g2):
            uf.union(t1, t2)

    # Assign compact new IDs
    root_to_new = {}
    next_id = [1]

    def get_new_id(root):
        if root not in root_to_new:
            root_to_new[root] = next_id[0]
            next_id[0] += 1
        return root_to_new[root]

    return {tid: get_new_id(uf.find(tid)) for tid in team_tids}


# ── public API ────────────────────────────────────────────────────────────────

def merge_player_tracks(
    tracks: dict,
    max_players_per_team: int = 11,
    position_threshold: float = DEFAULT_POSITION_THRESHOLD,
    min_frames: int = DEFAULT_MIN_FRAMES,
) -> dict:
    """
    Reduce player-ID fragmentation by linking track segments that belong to
    the same physical player.

    Must be called **after** ``team`` and ``position_transformed`` (or at
    minimum ``position_adjusted``) have been written into ``tracks["players"]``.

    Parameters
    ----------
    tracks : dict
        Full tracks dict from the pipeline.
    max_players_per_team : int
        Soft cap.  A warning is printed when more distinct IDs remain after
        linking (indicates the threshold may need adjustment).
    position_threshold : float
        Maximum pitch distance (metres) between the smoothed last position of
        one segment and the smoothed first position of the next to be
        considered a valid link.
    min_frames : int
        Segments shorter than this are dropped (likely false-positive
        detections) before linking.

    Returns
    -------
    dict – the mutated ``tracks`` dict with reassigned player IDs.
    """
    segs = _build_segments(tracks)

    # Discard very short / unteamed segments
    valid = {
        tid: seg for tid, seg in segs.items()
        if seg['frame_count'] >= min_frames and seg['team'] in (1, 2)
    }

    team_tids: dict = {1: [], 2: []}
    for tid, seg in valid.items():
        team_tids[seg['team']].append(tid)

    # Per-team remappings: {old_id: team-local_new_id}
    team_remaps: dict = {}
    team_offsets = {1: 0, 2: 100}   # team-2 IDs start at 101 to avoid clashes

    for team_id in (1, 2):
        tids = team_tids[team_id]
        if not tids:
            continue
        local_remap = _link_segments(valid, tids, position_threshold)
        offset = team_offsets[team_id]
        for old_tid, local_id in local_remap.items():
            team_remaps[old_tid] = local_id + offset

        n_final = len(set(local_remap.values()))
        n_orig = len(tids)
        print(f"[TrackMerger] Team {team_id}: {n_orig} segments → {n_final} identities", end='')
        if n_final > max_players_per_team:
            print(f"  ⚠  ({n_final} > {max_players_per_team}: consider raising position_threshold)")
        else:
            print()

    # Apply remapping to every frame
    for frame_num, frame_players in enumerate(tracks['players']):
        new_frame: dict = {}
        for old_tid, info in frame_players.items():
            new_tid = team_remaps.get(old_tid, old_tid)
            new_frame[new_tid] = info
        tracks['players'][frame_num] = new_frame

    total_before = len(valid)
    total_after = len(set(team_remaps.values()))
    print(f"[TrackMerger] Total: {total_before} segments → {total_after} identities")

    return tracks

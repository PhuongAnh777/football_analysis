"""
Post-processing step to reduce player-ID fragmentation produced by BoT-SORT.

When a camera cuts or pans hard, the tracker loses track IDs and issues fresh
ones.  This module links those short "track segments" back into a consistent
set of ≤22 identities (≤11 per team) by solving a small track-linking problem
per team.

Algorithm
---------
1. Build *segments*: for each unique track_id, record first/last frame, the
   smoothed first/last position on the pitch, team ID, and an averaged
   appearance feature vector from the player crop edges.
2. For each team, enumerate all *candidate links* (segment_A → segment_B) where
   - B starts after A ends  (no temporal overlap)
   - The combined score (spatial distance penalised by appearance similarity)
     is below the threshold.
3. Sort candidate links by score (best first) and use a **Union-Find** data
   structure to greedily merge them, verifying at each step that the merged
   group has no internal temporal conflict.
4. Assign new sequential IDs and rewrite every entry in ``tracks["players"]``.

Appearance ReID
---------------
When ``tracks["players"][f][tid]["appearance"]`` feature vectors are present
(added by ``Tracker.add_appearance_to_tracks``), the cosine similarity between
the L2-normalised 576-dim MobileNetV3 vectors at segment edges is used to
weight the spatial distance:

    effective_dist = spatial_dist × (1 − appearance_weight × similarity)

This means visually-identical segments get promoted in the merge priority even
if they are slightly farther apart spatially, which is critical across camera
cuts where players have moved while off-screen.

Call this function **after** team assignment and position transformation have
been added to ``tracks`` (i.e. near the end of ``main.py``).
"""

import numpy as np

# ── tuneable constants ────────────────────────────────────────────────────────

# Frames to average at the start/end of a segment for stable position/appearance.
# Increased from 5 → 10: more frames → smoother, more representative edge features.
_EDGE_SMOOTH_FRAMES = 10

# Default max spatial gap (metres on the transformed pitch) to consider two
# segments as candidates for linking.
# Increased from 30 → 40 m: allows linking across longer camera-cut gaps where
# the player has moved further while off-screen.
DEFAULT_POSITION_THRESHOLD = 40.0

# Segments shorter than this are discarded as detection noise before linking.
# Kept at 3: short segments from brief occlusions should still be kept.
DEFAULT_MIN_FRAMES = 3

# How strongly appearance similarity influences the effective-distance score.
# Increased from 0.7 → 0.8: jersey colour/texture is a very reliable cue in
# football; letting it dominate more aggressively reduces spatial false merges
# and recovers same-player links after camera cuts.
DEFAULT_APPEARANCE_WEIGHT = 0.8

# Maximum number of frames allowed between the end of one segment and the start
# of the next for them to be considered the same player.  Beyond this gap the
# player has likely been substituted, so we stop trying to link them.
# 300 frames ≈ 12.5 s @ 24 fps.
DEFAULT_MAX_GAP_FRAMES = 300


# ── appearance helpers ────────────────────────────────────────────────────────

def _cosine_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Cosine similarity ∈ [−1, 1] (in practice ≥ 0 for ReID features).
    Vectors are assumed L2-normalised; falls back to explicit normalisation.
    Returns 0.5 (neutral) when either vector is None or zero-norm.
    """
    if f1 is None or f2 is None:
        return 0.5
    n1, n2 = np.linalg.norm(f1), np.linalg.norm(f2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.5
    return float(np.dot(f1 / n1, f2 / n2))


def _mean_feat(feats: list):
    """Return element-wise mean of a list of feature arrays, or None."""
    valid = [f for f in feats if f is not None]
    if not valid:
        return None
    mean = np.mean(valid, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-9 else mean


# ── position helpers ──────────────────────────────────────────────────────────

def _best_position(info: dict):
    """Return the transformed pitch position (metres) from a track-info dict.

    Only ``position_transformed`` is used – pixel-space fallback is intentionally
    omitted because mixing pixel distances with the metre-scale threshold causes
    the merger to skip almost all candidate links.
    """
    pos = info.get('position_transformed')
    if pos is not None:
        return np.array(pos, dtype=float)
    return None


# ── segment builder ───────────────────────────────────────────────────────────

def _build_segments(tracks: dict) -> dict:
    """
    Scan all frames and build a segment descriptor per unique track_id.

    Returns
    -------
    dict mapping track_id → {
        'first_frame', 'last_frame',
        'first_positions'  : list (up to _EDGE_SMOOTH_FRAMES),
        'last_positions'   : list (up to _EDGE_SMOOTH_FRAMES),
        'first_appearances': list of feature arrays at segment start,
        'last_appearances' : list of feature arrays at segment end,
        'first_pos', 'last_pos', 'first_app', 'last_app',
        'frame_count', 'team'
    }
    """
    segs: dict = {}

    for frame_num, frame_players in enumerate(tracks['players']):
        for track_id, info in frame_players.items():
            pos = _best_position(info)
            app = info.get('appearance')  # np.ndarray or None

            if track_id not in segs:
                segs[track_id] = {
                    'first_frame':       frame_num,
                    'last_frame':        frame_num,
                    'first_positions':   [],
                    'last_positions':    [],
                    'first_appearances': [],
                    'last_appearances':  [],
                    'frame_count':       0,
                    'team':              info.get('team'),
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
                if len(seg['last_positions']) > _EDGE_SMOOTH_FRAMES:
                    seg['last_positions'].pop(0)

            if app is not None:
                if len(seg['first_appearances']) < _EDGE_SMOOTH_FRAMES:
                    seg['first_appearances'].append(app)
                seg['last_appearances'].append(app)
                if len(seg['last_appearances']) > _EDGE_SMOOTH_FRAMES:
                    seg['last_appearances'].pop(0)

    for seg in segs.values():
        seg['first_pos'] = (np.mean(seg['first_positions'], axis=0)
                            if seg['first_positions'] else None)
        seg['last_pos']  = (np.mean(seg['last_positions'],  axis=0)
                            if seg['last_positions']  else None)
        seg['first_app'] = _mean_feat(seg['first_appearances'])
        seg['last_app']  = _mean_feat(seg['last_appearances'])

    return segs


# ── Union-Find ────────────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self, keys):
        self._parent  = {k: k  for k in keys}
        self._rank    = {k: 0  for k in keys}
        self._members = {k: [k] for k in keys}

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
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        self._members[ra] = self._members[ra] + self._members[rb]
        self._members[rb] = self._members[ra]

    def groups(self):
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
                   position_threshold: float,
                   appearance_weight: float,
                   max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES) -> dict:
    """
    Return a remapping {old_track_id: new_id} for one team's track segments.
    Uses Union-Find with greedy candidate ordering (lowest effective score first).

    Effective score:
        effective_dist = spatial_dist × (1 − appearance_weight × cosine_sim)

    A high appearance similarity lowers the effective distance, promoting
    visually-consistent pairs even when the spatial gap is moderate (common
    after a camera cut where the player moved while off-screen).
    """
    uf = _UnionFind(team_tids)

    # Pure-appearance threshold: link two segments when their feature vectors
    # are very similar even if no pitch position is available.
    # Lowered from 0.80 → 0.75: slightly more permissive so that same-player
    # segments separated by a camera cut are still linked on appearance alone.
    _APPEARANCE_ONLY_SIM_THRESHOLD = 0.75

    candidates = []
    for i, t1 in enumerate(team_tids):
        for t2 in team_tids[i + 1:]:
            s1, s2 = segs[t1], segs[t2]

            if s1['last_frame'] < s2['first_frame']:
                earlier, later = s1, s2
            elif s2['last_frame'] < s1['first_frame']:
                earlier, later = s2, s1
            else:
                continue  # temporal overlap → cannot link

            # Skip pairs whose temporal gap exceeds the max-gap threshold.
            # A gap this large likely means a substitution rather than the
            # same player reappearing (camera cut gaps are usually < 5 s).
            frame_gap = later['first_frame'] - earlier['last_frame']
            if frame_gap > max_gap_frames:
                continue

            has_spatial = (earlier['last_pos'] is not None and
                           later['first_pos'] is not None)

            if has_spatial:
                spatial_dist = float(np.linalg.norm(
                    earlier['last_pos'] - later['first_pos']))
                app_sim = _cosine_similarity(earlier['last_app'], later['first_app'])
                effective_dist = spatial_dist * (1.0 - appearance_weight * app_sim)

                if effective_dist < position_threshold:
                    candidates.append((effective_dist, t1, t2))

            elif appearance_weight > 0:
                # No valid pitch position on one or both ends – fall back to
                # appearance-only linking with a stricter similarity gate.
                app_sim = _cosine_similarity(earlier['last_app'], later['first_app'])
                if app_sim >= _APPEARANCE_ONLY_SIM_THRESHOLD:
                    score = 1.0 - app_sim + 0.5
                    candidates.append((score, t1, t2))

    candidates.sort()

    for _, t1, t2 in candidates:
        r1, r2 = uf.find(t1), uf.find(t2)
        if r1 == r2:
            continue
        g1 = uf.members(t1)
        g2 = uf.members(t2)
        if not _temporal_conflict(segs, g1, g2):
            uf.union(t1, t2)

    root_to_new: dict = {}
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
    appearance_weight: float = DEFAULT_APPEARANCE_WEIGHT,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
) -> dict:
    """
    Reduce player-ID fragmentation by linking track segments that belong to
    the same physical player.

    Must be called **after** ``team`` and ``position_transformed`` (or at
    minimum ``position_adjusted``) have been written into ``tracks["players"]``.
    Optionally uses ``appearance`` feature vectors if present (added by
    ``Tracker.add_appearance_to_tracks``).

    Parameters
    ----------
    tracks : dict
        Full tracks dict from the pipeline.
    max_players_per_team : int
        Soft cap; a warning is printed when more distinct IDs remain after
        linking.
    position_threshold : float
        Gate on the *effective* (appearance-adjusted) spatial distance in
        metres.
    min_frames : int
        Segments shorter than this are dropped (likely false-positive
        detections) before linking.
    appearance_weight : float
        Weight of the appearance similarity term in [0, 1].  0 = spatial-only;
        1 = appearance fully cancels spatial distance for identical-looking
        segments.
    max_gap_frames : int
        Maximum frame gap between two segments for them to be considered the
        same player.  Prevents linking a player to a substitute who wears the
        same jersey number but enters the pitch much later.

    Returns
    -------
    dict – the mutated ``tracks`` dict with reassigned player IDs.
    """
    segs = _build_segments(tracks)

    valid = {
        tid: seg for tid, seg in segs.items()
        if seg['frame_count'] >= min_frames and seg['team'] in (1, 2)
    }

    team_tids: dict = {1: [], 2: []}
    for tid, seg in valid.items():
        team_tids[seg['team']].append(tid)

    has_appearance = any(
        seg.get('first_app') is not None for seg in valid.values()
    )
    n_with_pos = sum(1 for s in valid.values() if s.get('first_pos') is not None)
    print(f"[TrackMerger] {len(valid)} valid segments | "
          f"with pitch position: {n_with_pos} | "
          f"appearance: {'yes' if has_appearance else 'no'} (weight={appearance_weight})")

    team_remaps: dict = {}
    team_offsets = {1: 0, 2: 100}

    for team_id in (1, 2):
        tids = team_tids[team_id]
        if not tids:
            continue
        local_remap = _link_segments(
            valid, tids, position_threshold,
            appearance_weight if has_appearance else 0.0,
            max_gap_frames=max_gap_frames,
        )
        offset = team_offsets[team_id]
        for old_tid, local_id in local_remap.items():
            team_remaps[old_tid] = local_id + offset

        n_final = len(set(local_remap.values()))
        n_orig  = len(tids)
        print(f"[TrackMerger] Team {team_id}: {n_orig} segments → {n_final} identities", end='')
        if n_final > max_players_per_team:
            print(f"  ⚠  ({n_final} > {max_players_per_team}: consider raising position_threshold)")
        else:
            print()

    for frame_num, frame_players in enumerate(tracks['players']):
        new_frame: dict = {}
        for old_tid, info in frame_players.items():
            new_tid = team_remaps.get(old_tid, old_tid)
            new_frame[new_tid] = info
        tracks['players'][frame_num] = new_frame

    total_before = len(valid)
    total_after  = len(set(team_remaps.values()))
    print(f"[TrackMerger] Total: {total_before} segments → {total_after} identities")

    return tracks

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
4. **Force-cap phase**: if more than max_players groups remain, aggressively
   merge the smallest group into the nearest non-conflicting group (no
   distance threshold) until the cap is reached.
5. **Hard-prune phase**: keep only the top max_players groups by total frame
   count; all other track IDs are removed from the tracks dict entirely.
6. Assign new sequential IDs and rewrite every entry in ``tracks["players"]``.

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
_EDGE_SMOOTH_FRAMES = 10

# Default max spatial gap (metres on the transformed pitch) to consider two
# segments as candidates for linking.
DEFAULT_POSITION_THRESHOLD = 40.0

# Segments shorter than this are discarded as detection noise before linking.
DEFAULT_MIN_FRAMES = 3

# How strongly appearance similarity influences the effective-distance score.
DEFAULT_APPEARANCE_WEIGHT = 0.8

# Maximum number of frames allowed between the end of one segment and the start
# of the next for them to be considered the same player.
# Increased from 300 → 600: covers camera cuts up to ~25 s at 24 fps, which is
# more realistic for broadcast footage with replays and stoppages.
DEFAULT_MAX_GAP_FRAMES = 600

# Appearance-only link threshold (when no pitch position is available).
# Lowered from 0.75 → 0.62: jersey colour / texture under varying lighting
# often drops below 0.75, causing same-player segments to be missed.
_APPEARANCE_ONLY_SIM_THRESHOLD = 0.62


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


def _pair_score(segs: dict, g_a: list, g_b: list, appearance_weight: float) -> float:
    """
    Compute the minimum effective score across all cross-group endpoint pairs.
    Used by the force-cap phase where no distance threshold is applied.
    """
    best = float('inf')
    for ta in g_a:
        for tb in g_b:
            sa, sb = segs[ta], segs[tb]
            if sa['last_frame'] < sb['first_frame']:
                earlier, later = sa, sb
            elif sb['last_frame'] < sa['first_frame']:
                earlier, later = sb, sa
            else:
                continue  # segments overlap in time within the group

            has_spatial = (earlier['last_pos'] is not None and
                           later['first_pos'] is not None)
            if has_spatial:
                dist = float(np.linalg.norm(
                    earlier['last_pos'] - later['first_pos']))
                sim = _cosine_similarity(earlier['last_app'], later['first_app'])
                score = dist * (1.0 - appearance_weight * sim)
            else:
                sim = _cosine_similarity(
                    earlier.get('last_app'), later.get('first_app'))
                score = max(0.0, 1.5 - sim)
            best = min(best, score)
    return best


def _force_cap(
    segs: dict,
    uf: _UnionFind,
    cap: int,
    appearance_weight: float,
) -> None:
    """
    Phase 2: aggressively merge excess groups until at most `cap` remain.

    After the threshold-gated greedy phase, if more groups exist than the
    player cap, this function repeatedly merges the smallest group (fewest
    total frames) into the nearest non-temporally-conflicting group.
    No distance threshold is applied — the best available partner is always
    chosen.  If the smallest group conflicts with every other group (i.e. it
    represents a genuinely concurrent player or a substitution on the bench),
    the loop stops; the remaining excess will be handled by the hard-prune step.
    """
    def _group_frames(group):
        return sum(segs[tid]['frame_count'] for tid in group)

    for _ in range(5000):
        groups = uf.groups()
        if len(groups) <= cap:
            break

        groups_by_frames = sorted(groups, key=_group_frames)
        excess = groups_by_frames[0]

        best_score = float('inf')
        best_partner_tid = None

        for candidate in groups_by_frames[1:]:
            if _temporal_conflict(segs, excess, candidate):
                continue
            score = _pair_score(segs, excess, candidate, appearance_weight)
            if score < best_score:
                best_score = score
                best_partner_tid = candidate[0]

        if best_partner_tid is None:
            # This group temporally overlaps with every other group.
            # It is a genuinely concurrent identity (e.g. a substituting player).
            # The hard-prune step will decide whether to keep it.
            break

        uf.union(excess[0], best_partner_tid)


def _link_segments(segs: dict, team_tids: list,
                   position_threshold: float,
                   appearance_weight: float,
                   max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
                   max_players: int = 11) -> tuple:
    """
    Return ``(remap, drop_set)`` for one team's track segments.

    remap    : {old_track_id: new_local_id} for tracks that are kept.
    drop_set : set of old_track_ids to delete from the tracks dict entirely.

    Three phases:
    1. Greedy threshold-gated linking via Union-Find.
    2. Force-cap: merge smallest groups until ≤ max_players remain (or stuck).
    3. Hard-prune: keep top max_players groups by total frame count; build
       drop_set from the rest.
    """
    uf = _UnionFind(team_tids)

    # ── Phase 1: greedy threshold-gated linking ───────────────────────────────
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

    # ── Phase 2: force-cap ────────────────────────────────────────────────────
    if len(uf.groups()) > max_players:
        _force_cap(segs, uf, max_players, appearance_weight)

    # ── Phase 3: hard-prune — keep top max_players by total frame count ───────
    def _group_frames(group):
        return sum(segs[tid]['frame_count'] for tid in group)

    groups_sorted = sorted(uf.groups(), key=_group_frames, reverse=True)
    keep_groups = groups_sorted[:max_players]
    drop_groups = groups_sorted[max_players:]

    drop_set = {tid for g in drop_groups for tid in g}

    # Assign new sequential IDs (1-based) ordered by frame count descending
    remap: dict = {}
    for new_id, group in enumerate(keep_groups, start=1):
        for tid in group:
            remap[tid] = new_id

    return remap, drop_set


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
    the same physical player, then hard-cap each team to exactly
    ``max_players_per_team`` identities.

    Must be called **after** ``team`` and ``position_transformed`` (or at
    minimum ``position_adjusted``) have been written into ``tracks["players"]``.
    Optionally uses ``appearance`` feature vectors if present (added by
    ``Tracker.add_appearance_to_tracks``).

    Parameters
    ----------
    tracks : dict
        Full tracks dict from the pipeline.
    max_players_per_team : int
        Hard cap on the number of distinct player IDs kept per team.
        After greedy + force merging, the top identities by total frame count
        are kept and the rest are removed from the tracks entirely.
    position_threshold : float
        Gate on the *effective* (appearance-adjusted) spatial distance in
        metres for the greedy phase.
    min_frames : int
        Segments shorter than this are dropped (likely false-positive
        detections) before linking.
    appearance_weight : float
        Weight of the appearance similarity term in [0, 1].
    max_gap_frames : int
        Maximum frame gap between two segments for the greedy phase.

    Returns
    -------
    dict – the mutated ``tracks`` dict with reassigned and pruned player IDs.
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
    global_drop: set = set()
    team_offsets = {1: 0, 2: 100}

    for team_id in (1, 2):
        tids = team_tids[team_id]
        if not tids:
            continue
        local_remap, local_drop = _link_segments(
            valid, tids, position_threshold,
            appearance_weight if has_appearance else 0.0,
            max_gap_frames=max_gap_frames,
            max_players=max_players_per_team,
        )
        offset = team_offsets[team_id]
        for old_tid, local_id in local_remap.items():
            team_remaps[old_tid] = local_id + offset
        global_drop.update(local_drop)

        n_final = len(set(local_remap.values()))
        n_orig  = len(tids)
        dropped = len(local_drop)
        print(f"[TrackMerger] Team {team_id}: {n_orig} segments → "
              f"{n_final} identities (dropped {dropped} excess segments)")

    # Rewrite frame dicts: apply remap and remove pruned IDs
    for frame_num, frame_players in enumerate(tracks['players']):
        new_frame: dict = {}
        for old_tid, info in frame_players.items():
            if old_tid in global_drop:
                continue  # remove ghost / excess track
            new_tid = team_remaps.get(old_tid, old_tid)
            new_frame[new_tid] = info
        tracks['players'][frame_num] = new_frame

    total_before = len(valid)
    total_after  = len(set(team_remaps.values()))
    print(f"[TrackMerger] Total: {total_before} segments → {total_after} identities "
          f"(dropped {len(global_drop)} excess segments from tracks)")

    return tracks

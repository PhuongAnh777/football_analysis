"""
Tactical metrics derived from the position-transformed player tracks.

All functions operate on the ``tracks`` dict produced by the pipeline after:
  • position_transformed has been populated  (view_transformer)
  • team has been assigned                   (team_assigner)
  • speed has been populated                 (speed_and_distance_estimator)
  • has_ball has been assigned               (player_ball_assigner)

Coordinate system (from ViewTransformer):
  x  ∈ [0, 23.32] m  — along the pitch length (attacking direction per team)
  y  ∈ [0, 68]    m  — across the pitch width
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# ── pitch constants (from ViewTransformer target_vertices) ────────────────────
_PITCH_X_MAX = 23.32   # metres
_PITCH_Y_MAX = 68.0    # metres
_ATTACK_DIRECTION_FRAMES = 100   # frames used to infer team's attacking half


# ── private helpers ───────────────────────────────────────────────────────────

def _get_team_positions(tracks: dict, team_id, frame_idx: int) -> list:
    """Return list of [x, y] transformed positions for *team_id* in *frame_idx*."""
    result = []
    for info in tracks["players"][frame_idx].values():
        if info.get("team") == team_id:
            pos = info.get("position_transformed")
            if pos is not None:
                result.append(list(pos))
    return result


def _infer_attack_direction(tracks: dict, team_id) -> int:
    """
    Return +1 if the team attacks in the *increasing* x direction, -1 otherwise.

    Teams spending more time in the lower half of the visible x-range are
    assumed to be defending there, i.e. attacking towards higher x.
    """
    xs = []
    for frame in tracks["players"][:_ATTACK_DIRECTION_FRAMES]:
        for info in frame.values():
            if info.get("team") == team_id:
                pos = info.get("position_transformed")
                if pos is not None:
                    xs.append(pos[0])
    if not xs:
        return +1
    return +1 if np.mean(xs) < _PITCH_X_MAX / 2 else -1


def _cluster_players_by_line(x_vals: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Cluster 1-D x-positions (along attacking axis) into 2–4 tactical lines.

    Returns
    -------
    labels   : np.ndarray of int cluster labels (sorted 0=defensive, N=attacking)
    n_clusters : int
    centers  : np.ndarray of cluster centres (sorted ascending)
    """
    n = len(x_vals)
    if n < 2:
        return np.zeros(n, dtype=int), 1, x_vals.copy()

    best_k, best_score = 2, -2.0
    for k in range(2, min(5, n)):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(x_vals.reshape(-1, 1))
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(x_vals.reshape(-1, 1), labels)
        if score > best_score:
            best_k, best_score = k, score

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    raw_labels = km.fit_predict(x_vals.reshape(-1, 1))
    centers = km.cluster_centers_.flatten()

    # Remap so label 0 = most defensive cluster (lowest attacking-x)
    order = np.argsort(centers)
    remap = {old: new for new, old in enumerate(order)}
    sorted_labels = np.array([remap[l] for l in raw_labels])
    sorted_centers = np.sort(centers)

    return sorted_labels, best_k, sorted_centers


# ── public API ────────────────────────────────────────────────────────────────

def detect_formation(tracks: dict, team_id, frame_idx: int) -> str:
    """
    Detect the tactical formation of *team_id* in *frame_idx*.

    Clusters players into 2–4 lines along the attacking axis using KMeans with
    silhouette-score selection.  The goalkeeper (singleton at the defensive end)
    is stripped before formatting the string.

    Parameters
    ----------
    tracks   : full pipeline tracks dict
    team_id  : team identifier (as stored in tracks["players"][...][id]["team"])
    frame_idx: frame index to analyse

    Returns
    -------
    str  e.g. "4-3-3", "4-4-2", "3-5-2", or "unknown" if < 4 players visible
    """
    positions = _get_team_positions(tracks, team_id, frame_idx)
    if len(positions) < 4:
        return "unknown"

    pts = np.array(positions)
    attack_dir = _infer_attack_direction(tracks, team_id)
    x_vals = pts[:, 0] * attack_dir   # flip so "forward" is always increasing

    labels, n_clusters, _ = _cluster_players_by_line(x_vals)

    counts = [int(np.sum(labels == k)) for k in range(n_clusters)]

    # Strip goalkeeper: defensive-most singleton cluster
    if counts[0] == 1 and n_clusters > 1:
        counts = counts[1:]

    return "-".join(str(c) for c in counts)


def compute_compact_score(tracks: dict, team_id, frame_idx: int):
    """
    Mean pairwise distance (metres) between all visible players of *team_id*.

    Lower score = more compact shape.

    Returns
    -------
    float, or None if fewer than 2 players are visible.
    """
    positions = _get_team_positions(tracks, team_id, frame_idx)
    if len(positions) < 2:
        return None

    pts = np.array(positions)
    dists = cdist(pts, pts, metric="euclidean")
    n = len(pts)
    upper = dists[np.triu_indices(n, k=1)]
    return float(upper.mean())


def compute_pressing_intensity(
    tracks: dict,
    team_id,
    frame_idx: int,
    radius_m: float = 10.0,
) -> float:
    """
    Fraction of opponents within *radius_m* metres of the ball carrier.

    Returns
    -------
    float in [0.0, 1.0].  Returns 0.0 if ball carrier or opponents are absent.
    """
    frame = tracks["players"][frame_idx]

    # Locate the ball carrier belonging to team_id
    ball_pos = None
    for info in frame.values():
        if info.get("has_ball") and info.get("team") == team_id:
            pos = info.get("position_transformed")
            if pos is not None:
                ball_pos = np.array(pos)
                break

    if ball_pos is None:
        return 0.0

    opponents = [
        info for info in frame.values()
        if info.get("team") != team_id
        and info.get("position_transformed") is not None
    ]
    if not opponents:
        return 0.0

    near = sum(
        1 for opp in opponents
        if np.linalg.norm(np.array(opp["position_transformed"]) - ball_pos) <= radius_m
    )
    return near / len(opponents)


def compute_team_speed(tracks: dict, team_id, frame_idx: int):
    """
    Average speed (km/h) of all *team_id* players in *frame_idx*.

    Returns
    -------
    float, or None if no speed data is available for the team in that frame.
    """
    frame = tracks["players"][frame_idx]
    speeds = [
        info["speed"]
        for info in frame.values()
        if info.get("team") == team_id and info.get("speed") is not None
    ]
    return float(np.mean(speeds)) if speeds else None


def analyze_all_frames(tracks: dict, sample_every: int = 30) -> dict:
    """
    Run all four tactical metrics every *sample_every* frames.

    Automatically discovers the team IDs present in *tracks*.

    Parameters
    ----------
    tracks       : full pipeline tracks dict (must have team, position_transformed,
                   speed, and has_ball populated)
    sample_every : stride between analysed frames

    Returns
    -------
    dict keyed by team_id, each value is a list of dicts::

        {
            "frame":     int,
            "formation": str,
            "compact":   float | None,
            "pressing":  float,
            "avg_speed": float | None,
        }
    """
    # Discover all team IDs present in the data
    team_ids: set = set()
    for frame in tracks["players"]:
        for info in frame.values():
            t = info.get("team")
            if t is not None:
                team_ids.add(t)

    results: dict = {t: [] for t in sorted(team_ids)}
    n_frames = len(tracks["players"])

    for frame_idx in range(0, n_frames, sample_every):
        for team_id in sorted(team_ids):
            results[team_id].append({
                "frame":     frame_idx,
                "formation": detect_formation(tracks, team_id, frame_idx),
                "compact":   compute_compact_score(tracks, team_id, frame_idx),
                "pressing":  compute_pressing_intensity(tracks, team_id, frame_idx),
                "avg_speed": compute_team_speed(tracks, team_id, frame_idx),
            })

    return results

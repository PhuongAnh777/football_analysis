import matplotlib
matplotlib.use('Agg')  # non-interactive backend — required when running in threads
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import math

# Full FIFA pitch dimensions (IFAB Law 1, international matches)
PITCH_LENGTH = 105.0   # goal line → goal line (metres) — pos[0] axis
PITCH_WIDTH  =  68.0   # touchline → touchline  (metres) — pos[1] axis

# Maximum frames between two possession events to still count as a pass
MAX_PASS_GAP_FRAMES = 50

# Ignore transformed positions outside the pitch (bad calibration / camera-offset noise)
_PITCH_MARGIN_M = 1.0
_MIN_POSITION_SAMPLES = 10

_PENALTY_SPOT_X = 11.0
_PENALTY_ARC_R  =  9.15
_PENALTY_ARC_ANG = math.degrees(math.acos(
    (16.5 - _PENALTY_SPOT_X) / _PENALTY_ARC_R
))  # ≈ 53.1°


def _bgr_to_rgb_float(bgr_color):
    b, g, r = bgr_color
    return (r / 255.0, g / 255.0, b / 255.0)


def _draw_pitch(ax, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """Draw a full FIFA-standard pitch on *ax*.

    Orientation (landscape):
        x-axis  = along pitch LENGTH (pos[0], 0 → 105 m)
        y-axis  = across pitch WIDTH (pos[1], 0 →  68 m)

    Markings drawn:
        • Boundary, halfway line, center spot
        • Center circle (r = 9.15 m)
        • Penalty areas, goal areas, penalty spots
        • Penalty D arcs
        • Corner arcs (r = 1 m)
    """
    ax.set_facecolor('#2d5a1b')

    # alternating length-wise stripes
    stripe_l = length / 10
    for i in range(10):
        c = '#2d5a1b' if i % 2 == 0 else '#347322'
        ax.add_patch(patches.Rectangle(
            (i * stripe_l, 0), stripe_l, width, facecolor=c, zorder=0))

    mid_y = width / 2   # 34 m

    # ── boundary ────────────────────────────────────────────────────────
    ax.add_patch(patches.Rectangle(
        (0, 0), length, width, fill=False,
        edgecolor='white', linewidth=2.0, zorder=1))

    # ── halfway line ─────────────────────────────────────────────────────
    ax.plot([length / 2, length / 2], [0, width],
            color='white', linewidth=1.5, zorder=1)

    # ── center circle ────────────────────────────────────────────────────
    ax.add_patch(patches.Circle(
        (length / 2, mid_y), 9.15, fill=False,
        edgecolor='white', linewidth=1.5, zorder=1))

    # ── center spot ──────────────────────────────────────────────────────
    ax.scatter([length / 2], [mid_y], s=25, c='white', zorder=2)

    # ── penalty & goal areas, spots, D arcs (both ends) ─────────────────
    for side in ('left', 'right'):
        # penalty area (16.5m × 40.32m)
        x_pen = 0 if side == 'left' else length - 16.5
        ax.add_patch(patches.Rectangle(
            (x_pen, mid_y - 20.16), 16.5, 40.32,
            fill=False, edgecolor='white', linewidth=1.5, zorder=1))

        # goal area (5.5m × 18.32m)
        x_goal = 0 if side == 'left' else length - 5.5
        ax.add_patch(patches.Rectangle(
            (x_goal, mid_y - 9.16), 5.5, 18.32,
            fill=False, edgecolor='white', linewidth=1.5, zorder=1))

        # penalty spot
        spot_x = (_PENALTY_SPOT_X
                  if side == 'left' else length - _PENALTY_SPOT_X)
        ax.scatter([spot_x], [mid_y], s=20, c='white', zorder=2)

        # penalty D arc (portion outside the penalty area)
        ang    = np.linspace(
            np.radians(-_PENALTY_ARC_ANG),
            np.radians( _PENALTY_ARC_ANG),
            100
        )
        arc_x  = spot_x + _PENALTY_ARC_R * np.cos(ang)
        arc_y  = mid_y  + _PENALTY_ARC_R * np.sin(ang)
        mask   = arc_x > 16.5 if side == 'left' else arc_x < length - 16.5
        if mask.any():
            ax.plot(arc_x[mask], arc_y[mask],
                    color='white', linewidth=1.5, zorder=1)

    # ── corner arcs (r = 1 m) ───────────────────────────────────────────
    for cx, cy, t1, t2 in [
        (0,      0,     0,  90),
        (length, 0,    90, 180),
        (0,      width, -90,   0),
        (length, width, 180, 270),
    ]:
        theta = np.linspace(np.radians(t1), np.radians(t2), 30)
        ax.plot(cx + np.cos(theta), cy + np.sin(theta),
                color='white', linewidth=1.2, zorder=1)

    ax.set_xlim(-1, length + 1)
    ax.set_ylim(-1, width + 1)
    ax.set_aspect('equal')
    ax.axis('off')


def _is_on_pitch(x, y, length=PITCH_LENGTH, width=PITCH_WIDTH, margin=_PITCH_MARGIN_M):
    """Return True when (x, y) lies inside the pitch bounds (with small margin)."""
    return (
        -margin <= x <= length + margin
        and -margin <= y <= width + margin
    )


def _detect_passes(tracks):
    """Scan ``has_ball`` flags and return a list of (from_id, to_id, team) tuples.

    A pass is recorded when ball possession transfers between two players
    of the same team within MAX_PASS_GAP_FRAMES frames.
    """
    passes = []
    prev_player_id = None
    prev_team      = None
    frames_without = 0

    for frame_players in tracks['players']:
        cur_id   = None
        cur_team = None
        for player_id, info in frame_players.items():
            if info.get('has_ball', False):
                cur_id   = player_id
                cur_team = info.get('team')
                break

        if cur_id is not None:
            if (prev_player_id is not None
                    and cur_id != prev_player_id
                    and cur_team == prev_team
                    and frames_without <= MAX_PASS_GAP_FRAMES):
                passes.append((prev_player_id, cur_id, cur_team))
            prev_player_id = cur_id
            prev_team      = cur_team
            frames_without = 0
        else:
            frames_without += 1

    return passes


def _player_avg_positions(
    tracks,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
):
    """Return ``{player_id: (avg_x, avg_y)}`` for pitch plotting.

    avg_x = mean pos[0]  (along pitch length, 0–105 m) → x-axis.
    avg_y = mean pos[1]  (across pitch width,  0–68  m) → y-axis.

    Only in-bounds samples are averaged so camera-offset drift does not
    pull nodes and edge labels off the pitch.
    """
    sums = defaultdict(lambda: [0.0, 0.0, 0])
    for frame_players in tracks['players']:
        for player_id, info in frame_players.items():
            pos = info.get('position_transformed')
            if pos is None:
                continue
            x, y = float(pos[0]), float(pos[1])
            if not _is_on_pitch(x, y, pitch_length, pitch_width):
                continue
            sums[player_id][0] += x   # length
            sums[player_id][1] += y   # width
            sums[player_id][2] += 1

    return {
        pid: (v[0] / v[2], v[1] / v[2])
        for pid, v in sums.items()
        if v[2] >= _MIN_POSITION_SAMPLES
    }


def _player_teams(tracks):
    """Return ``{player_id: team_id}``."""
    result = {}
    for frame_players in tracks['players']:
        for player_id, info in frame_players.items():
            team = info.get('team')
            if team is not None:
                result[player_id] = team
    return result


def generate_passing_network(
    tracks,
    team_colors,
    output_path='output_videos/passing_network.png',
    team_id=None,
    pitch_length: float = PITCH_LENGTH,
    pitch_width:  float = PITCH_WIDTH,
):
    """Draw a passing network for each team on a FIFA 105 × 68 m pitch.

    Node size  ∝  pass involvement (sent + received).
    Edge width ∝  pass frequency between a pair of players.

    Parameters
    ----------
    tracks : dict
        Full pipeline ``tracks`` dict.
    team_colors : dict
        ``{team_id: (B, G, R)}`` from ``TeamAssigner.team_colors``.
    output_path : str
        Destination PNG path.
    pitch_length, pitch_width : float
        Pitch dimensions matching the ViewTransformer calibration.
    """
    passes       = _detect_passes(tracks)
    avg_positions = _player_avg_positions(
        tracks, pitch_length=pitch_length, pitch_width=pitch_width,
    )
    player_team_map = _player_teams(tracks)

    pass_counts: dict  = defaultdict(int)
    involvement: dict  = defaultdict(int)
    for from_p, to_p, team in passes:
        pass_counts[(from_p, to_p, team)] += 1
        involvement[from_p] += 1
        involvement[to_p]   += 1

    team_players: dict = defaultdict(set)
    for pid, tid in player_team_map.items():
        team_players[tid].add(pid)

    node_fallback = {1: '#e74c3c', 2: '#3498db'}
    teams_to_draw = [team_id] if team_id in (1, 2) else [1, 2]

    # Figure sized for landscape 105×68 pitch (aspect ≈ 1.54:1)
    fig_w_per = 15.0
    fig_h     = fig_w_per * pitch_width / pitch_length   # ≈ 9.7"
    fig, axes = plt.subplots(
        1, len(teams_to_draw),
        figsize=(fig_w_per * len(teams_to_draw), fig_h),
        squeeze=False,
    )
    fig.patch.set_facecolor('#111827')
    axes = axes[0]

    player_numbers = {
        pid: idx + 1
        for tid in teams_to_draw
        for idx, pid in enumerate(sorted(team_players.get(tid, [])))
    }

    for idx, draw_team_id in enumerate(teams_to_draw):
        ax = axes[idx]
        _draw_pitch(ax, length=pitch_length, width=pitch_width)

        players_in_team = team_players.get(draw_team_id, set())
        n_passes_team   = sum(
            1 for _, _, t in passes if t == draw_team_id
        )

        # team title inside the axes
        ax.text(
            pitch_length / 2, pitch_width + 0.3,
            f'Team {draw_team_id}  ·  {n_passes_team} passes detected',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='white', zorder=10,
        )

        # ── edges ────────────────────────────────────────────────────────
        edge_counts: dict = defaultdict(int)
        for (from_p, to_p, team), cnt in pass_counts.items():
            if team == draw_team_id:
                key = (min(from_p, to_p), max(from_p, to_p))
                edge_counts[key] += cnt

        max_edge = max(edge_counts.values(), default=1) or 1

        for (p1, p2), cnt in edge_counts.items():
            if p1 not in avg_positions or p2 not in avg_positions:
                continue
            x1, y1 = avg_positions[p1]
            x2, y2 = avg_positions[p2]
            if not (
                _is_on_pitch(x1, y1, pitch_length, pitch_width)
                and _is_on_pitch(x2, y2, pitch_length, pitch_width)
            ):
                continue
            lw    = 0.8 + 6.0 * (cnt / max_edge)
            alpha = 0.25 + 0.70 * (cnt / max_edge)
            ax.plot([x1, x2], [y1, y2],
                    color='white', linewidth=lw, alpha=alpha, zorder=2,
                    solid_capstyle='round')
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if _is_on_pitch(mx, my, pitch_length, pitch_width):
                ax.text(mx, my, str(cnt), ha='center', va='center',
                        fontsize=7, color='#111827', fontweight='bold', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  alpha=0.75, edgecolor='none'))

        # ── nodes ────────────────────────────────────────────────────────
        node_color = (_bgr_to_rgb_float(team_colors[draw_team_id])
                      if draw_team_id in team_colors
                      else node_fallback[draw_team_id])
        max_inv = max(
            (involvement[p] for p in players_in_team), default=1
        ) or 1

        for player_id in players_in_team:
            if player_id not in avg_positions:
                continue
            x, y  = avg_positions[player_id]
            if not _is_on_pitch(x, y, pitch_length, pitch_width):
                continue
            inv   = involvement.get(player_id, 0)
            size  = 180 + 700 * (inv / max_inv)
            ax.scatter(x, y, s=size, c=[node_color], zorder=4,
                       edgecolors='white', linewidths=1.8)
            ax.text(x, y, str(player_numbers.get(player_id, '?')),
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold', zorder=5)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualization] Passing network saved → {output_path}")

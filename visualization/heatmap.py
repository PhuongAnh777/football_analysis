import math
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — required when running in threads
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from collections import defaultdict

# Full FIFA pitch dimensions (IFAB Law 1, international matches)
PITCH_LENGTH = 105.0   # goal line → goal line (metres) — pos[0] axis
PITCH_WIDTH  =  68.0   # touchline → touchline  (metres) — pos[1] axis

COLS = 5   # players per row in the per-player grid

# Derived arc angle for penalty D
_PENALTY_SPOT_X  = 11.0        # penalty spot from goal line
_PENALTY_ARC_R   =  9.15       # radius of penalty D
_PENALTY_ARC_ANG = math.degrees(math.acos(
    (16.5 - _PENALTY_SPOT_X) / _PENALTY_ARC_R
))  # ≈ 53.1°


def _bgr_to_rgb_float(bgr_color):
    """Convert OpenCV BGR tuple to matplotlib (R, G, B) in [0, 1]."""
    b, g, r = bgr_color
    return (r / 255.0, g / 255.0, b / 255.0)


def _draw_mini_pitch(ax, length=PITCH_LENGTH, width=PITCH_WIDTH, detailed=False):
    """Draw a pitch background on *ax*.

    Orientation (landscape):
        x-axis  = along pitch LENGTH (pos[0], 0 → length m)
        y-axis  = across pitch WIDTH (pos[1], 0 → width  m)

    Parameters
    ----------
    detailed : bool
        When True draws penalty areas, center circle, etc.
        When False (default for small per-player subplots) draws only
        boundary, halfway line and center spot.
    """
    ax.set_facecolor('#2d5a1b')

    # alternating length-wise stripes
    n_stripes = 10 if detailed else 8
    stripe_l = length / n_stripes
    for i in range(n_stripes):
        c = '#2d5a1b' if i % 2 == 0 else '#326b1e'
        ax.add_patch(patches.Rectangle(
            (i * stripe_l, 0), stripe_l, width, facecolor=c, zorder=0))

    lw_main = 1.5 if detailed else 1.0

    # ── boundary ────────────────────────────────────────────────────────
    ax.add_patch(patches.Rectangle(
        (0, 0), length, width, fill=False,
        edgecolor='white', linewidth=lw_main, zorder=1))

    # ── halfway line ────────────────────────────────────────────────────
    ax.plot([length / 2, length / 2], [0, width],
            color='white', linewidth=lw_main, zorder=1)

    # ── center spot ─────────────────────────────────────────────────────
    ax.scatter([length / 2], [width / 2], s=15, c='white', zorder=2)

    if detailed:
        _draw_fifa_markings(ax, length, width)

    ax.set_xlim(-1, length + 1)
    ax.set_ylim(-1, width + 1)
    ax.set_aspect('equal')
    ax.axis('off')


def _draw_fifa_markings(ax, length, width):
    """Add FIFA-standard markings to an existing axes (no background)."""
    mid_y  = width / 2   # 34 m

    # ── center circle (r = 9.15 m) ──────────────────────────────────────
    ax.add_patch(patches.Circle(
        (length / 2, mid_y), 9.15, fill=False,
        edgecolor='white', linewidth=1.5, zorder=1))

    # ── penalty & goal areas, spots (both ends) ─────────────────────────
    for side in ('left', 'right'):
        x0 = 0 if side == 'left' else length - 16.5
        xg = 0 if side == 'left' else length - 5.5
        spot_x = _PENALTY_SPOT_X if side == 'left' else length - _PENALTY_SPOT_X

        # penalty area
        ax.add_patch(patches.Rectangle(
            (x0, mid_y - 20.16), 16.5, 40.32,
            fill=False, edgecolor='white', linewidth=1.5, zorder=1))
        # goal area
        ax.add_patch(patches.Rectangle(
            (xg, mid_y - 9.16), 5.5, 18.32,
            fill=False, edgecolor='white', linewidth=1.5, zorder=1))
        # penalty spot
        ax.scatter([spot_x], [mid_y], s=15, c='white', zorder=2)

        # penalty D arc (outside penalty area)
        ang = np.linspace(
            np.radians(-_PENALTY_ARC_ANG),
            np.radians( _PENALTY_ARC_ANG),
            80
        )
        arc_x = spot_x + _PENALTY_ARC_R * np.cos(ang)
        arc_y = mid_y  + _PENALTY_ARC_R * np.sin(ang)
        # keep only the part outside the penalty area
        if side == 'left':
            mask = arc_x > 16.5
        else:
            mask = arc_x < length - 16.5
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
        theta = np.linspace(np.radians(t1), np.radians(t2), 25)
        ax.plot(cx + np.cos(theta), cy + np.sin(theta),
                color='white', linewidth=1.2, zorder=1)


def _collect_player_data(tracks):
    """Return ``{player_id: [[pos0, pos1], ...]}`` and ``{player_id: team}``."""
    player_positions: dict = defaultdict(list)
    player_team: dict = {}

    for frame_players in tracks['players']:
        for player_id, info in frame_players.items():
            team = info.get('team')
            pos  = info.get('position_transformed')
            if team in (1, 2) and pos is not None:
                player_positions[player_id].append(pos)
                player_team[player_id] = team

    return player_positions, player_team


def _kde_heatmap(ax, positions, cmap, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """Overlay a KDE heatmap on *ax*.

    pos[0] → x-axis (along pitch length, 0–length m).
    pos[1] → y-axis (across pitch width,  0–width  m).
    """
    if len(positions) < 5:
        ax.text(length / 2, width / 2,
                'Not enough\ndata', ha='center', va='center',
                fontsize=6, color='white', zorder=3)
        return

    pts    = np.array(positions)
    x_plot = pts[:, 0]   # pos[0] = along pitch length
    y_plot = pts[:, 1]   # pos[1] = across pitch width

    # grid density proportional to pitch dimensions (≈2 pts per metre)
    n_x = max(40, int(length * 2))
    n_y = max(25, int(width  * 2))
    xx, yy = np.mgrid[0:length:complex(0, n_x), 0:width:complex(0, n_y)]
    grid   = np.vstack([xx.ravel(), yy.ravel()])
    try:
        kernel  = gaussian_kde(np.vstack([x_plot, y_plot]), bw_method=0.25)
        density = kernel(grid).reshape(xx.shape)
        ax.pcolormesh(xx, yy, density, cmap=cmap,
                      alpha=0.80, shading='auto', zorder=2)
    except np.linalg.LinAlgError:
        ax.scatter(x_plot, y_plot, c='yellow', s=5, alpha=0.5, zorder=3)


def generate_heatmap(
    tracks,
    team_colors,
    output_path='output_videos/heatmap.png',
    team_id=None,
    pitch_length: float = PITCH_LENGTH,
    pitch_width:  float = PITCH_WIDTH,
):
    """Generate a per-player KDE heatmap on a 105 × 68 m pitch.

    Players are grouped by team (Team 1 first, Team 2 below).
    Each subplot shows that player's KDE density overlaid on a mini pitch.

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
    player_positions, player_team = _collect_player_data(tracks)

    team_players: dict = {1: [], 2: []}
    for pid, tid in player_team.items():
        if pid in player_positions:
            team_players[tid].append(pid)

    teams_to_draw = [team_id] if team_id in (1, 2) else [1, 2]
    # Sort each team's players by average depth (pos[0]) → DEF first, FWD last
    for tid in teams_to_draw:
        team_players[tid].sort(
            key=lambda p: np.mean([pos[0] for pos in player_positions[p]])
        )

    cmap_per_team = {1: 'Reds', 2: 'Blues'}
    fallback_color = {1: (0.9, 0.3, 0.3), 2: (0.3, 0.6, 0.95)}

    rows_per_team = {
        tid: max(1, math.ceil(len(team_players[tid]) / COLS))
        for tid in teams_to_draw
    }
    include_headers = team_id not in (1, 2)
    height_ratios   = []
    for tid in teams_to_draw:
        if include_headers:
            height_ratios.append(0.25)
        height_ratios.extend([1.0] * rows_per_team[tid])

    # Cell height scaled to pitch aspect ratio (landscape: length > width)
    cell_w = 3.2                              # inches per column
    cell_h = cell_w * pitch_width / pitch_length   # e.g. 3.2 * 68/105 ≈ 2.07"

    fig_w = COLS * cell_w
    fig_h = sum(0.6 if r == 0.25 else cell_h for r in height_ratios)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='#111827')
    gs  = GridSpec(len(height_ratios), COLS,
                   figure=fig,
                   height_ratios=height_ratios,
                   hspace=0.55, wspace=0.10)

    current_row = 0

    def _team_label_row(row_idx, draw_team_id):
        ax_label = fig.add_subplot(gs[row_idx, :])
        ax_label.axis('off')
        color = (_bgr_to_rgb_float(team_colors[draw_team_id])
                 if draw_team_id in team_colors else fallback_color[draw_team_id])
        n = len(team_players[draw_team_id])
        ax_label.text(0.5, 0.5,
                      f'TEAM {draw_team_id}  ·  {n} players',
                      ha='center', va='center',
                      fontsize=14, fontweight='bold', color=color,
                      transform=ax_label.transAxes)
        ax_label.set_facecolor('#111827')
        ax_label.axhline(0.1, color=color, linewidth=1.5, alpha=0.4,
                         xmin=0.05, xmax=0.95)

    def _player_grid(start_row, draw_team_id):
        players = team_players[draw_team_id]
        cmap    = cmap_per_team[draw_team_id]
        color   = (_bgr_to_rgb_float(team_colors[draw_team_id])
                   if draw_team_id in team_colors else fallback_color[draw_team_id])

        for i, player_id in enumerate(players):
            row = start_row + i // COLS
            col = i % COLS
            ax  = fig.add_subplot(gs[row, col])
            _draw_mini_pitch(ax, length=pitch_length, width=pitch_width)
            _kde_heatmap(ax, player_positions[player_id], cmap,
                         length=pitch_length, width=pitch_width)

            n_frames = len(player_positions[player_id])
            ax.set_title(f'Cầu thủ {i + 1}  ({n_frames} frames)',
                         fontsize=7.5, color=color, fontweight='bold', pad=3)

        # empty cells
        total_cells = math.ceil(len(players) / COLS) * COLS
        for j in range(len(players), total_cells):
            row = start_row + j // COLS
            col = j % COLS
            ax_empty = fig.add_subplot(gs[row, col])
            ax_empty.axis('off')
            ax_empty.set_facecolor('#111827')

    for draw_team_id in teams_to_draw:
        if include_headers:
            _team_label_row(current_row, draw_team_id)
            current_row += 1
        _player_grid(current_row, draw_team_id)
        current_row += rows_per_team[draw_team_id]

    plt.savefig(output_path, dpi=140, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualization] Individual heatmap saved → {output_path}")

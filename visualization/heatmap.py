import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from collections import defaultdict

# Pitch dimensions matching ViewTransformer (meters)
PITCH_LENGTH = 23.32
PITCH_WIDTH = 68.0

COLS = 5  # players per row in the grid


def _bgr_to_rgb_float(bgr_color):
    """Convert OpenCV BGR tuple to matplotlib (R, G, B) in [0, 1]."""
    b, g, r = bgr_color
    return (r / 255.0, g / 255.0, b / 255.0)


def _draw_mini_pitch(ax, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """Draw a minimal pitch background on *ax*."""
    ax.set_facecolor('#2d5a1b')

    stripe_w = length / 8
    for i in range(8):
        c = '#2d5a1b' if i % 2 == 0 else '#326b1e'
        ax.add_patch(patches.Rectangle((i * stripe_w, 0), stripe_w, width,
                                       facecolor=c, zorder=0))

    ax.add_patch(patches.Rectangle((0, 0), length, width,
                                   fill=False, edgecolor='white',
                                   linewidth=1.2, zorder=1))
    ax.axvline(x=length / 2, color='white', linewidth=0.8, alpha=0.6, zorder=1)

    ax.set_xlim(-0.3, length + 0.3)
    ax.set_ylim(-0.3, width + 0.3)
    ax.set_aspect('equal')
    ax.axis('off')


def _collect_player_data(tracks):
    """
    Return two dicts:
    - player_positions : {player_id: [[x, y], ...]}
    - player_team      : {player_id: team_id}
    """
    player_positions: dict = defaultdict(list)
    player_team: dict = {}

    for frame_players in tracks['players']:
        for player_id, info in frame_players.items():
            team = info.get('team')
            pos = info.get('position_transformed')
            if team in (1, 2) and pos is not None:
                player_positions[player_id].append(pos)
                player_team[player_id] = team

    return player_positions, player_team


def _kde_heatmap(ax, positions, cmap):
    """Overlay a KDE heatmap on *ax* if there are enough points."""
    if len(positions) < 5:
        ax.text(PITCH_LENGTH / 2, PITCH_WIDTH / 2,
                'Not enough\ndata', ha='center', va='center',
                fontsize=6, color='white', zorder=3)
        return

    pts = np.array(positions)
    x, y = pts[:, 0], pts[:, 1]

    xx, yy = np.mgrid[0:PITCH_LENGTH:80j, 0:PITCH_WIDTH:150j]
    grid = np.vstack([xx.ravel(), yy.ravel()])
    try:
        kernel = gaussian_kde(np.vstack([x, y]), bw_method=0.35)
        density = kernel(grid).reshape(xx.shape)
        ax.pcolormesh(xx, yy, density, cmap=cmap,
                      alpha=0.80, shading='auto', zorder=2)
    except np.linalg.LinAlgError:
        # Singular matrix – all points collapsed to one location
        ax.scatter(x, y, c='yellow', s=5, alpha=0.5, zorder=3)


def generate_heatmap(tracks, team_colors, output_path='output_videos/heatmap.png'):
    """
    Generate an individual heatmap for every tracked outfield player.

    Players are grouped by team (Team 1 on top, Team 2 on bottom).
    Each subplot shows that player's KDE density on a mini pitch.

    Parameters
    ----------
    tracks : dict
        Full ``tracks`` dict from the main pipeline.
    team_colors : dict
        ``{team_id: (B, G, R)}`` from ``TeamAssigner.team_colors``.
    output_path : str
        Destination PNG path.
    """
    player_positions, player_team = _collect_player_data(tracks)

    # Group and sort players by team then by average x (field position)
    team_players: dict = {1: [], 2: []}
    for pid, tid in player_team.items():
        if pid in player_positions:
            team_players[tid].append(pid)

    for tid in (1, 2):
        team_players[tid].sort(
            key=lambda p: np.mean([pos[0] for pos in player_positions[p]])
        )

    cmap_per_team = {1: 'Reds', 2: 'Blues'}
    fallback_color = {1: (0.9, 0.3, 0.3), 2: (0.3, 0.6, 0.95)}

    # ── figure layout ──────────────────────────────────────────────────────
    # Each team occupies ceil(n_players / COLS) rows; teams are separated
    # by a blank "header" row for the team label.
    rows_t1 = max(1, math.ceil(len(team_players[1]) / COLS))
    rows_t2 = max(1, math.ceil(len(team_players[2]) / COLS))

    # Heights: header rows are thin (0.3), player rows are normal (1.0)
    height_ratios = ([0.25] + [1.0] * rows_t1 +
                     [0.25] + [1.0] * rows_t2)
    total_rows = 1 + rows_t1 + 1 + rows_t2

    fig_w = COLS * 3.2
    fig_h = sum(0.6 if r == 0.25 else 3.0
                for r in height_ratios)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='#111827')
    gs = GridSpec(total_rows, COLS,
                  figure=fig,
                  height_ratios=height_ratios,
                  hspace=0.45, wspace=0.15)

    def _team_label_row(row_idx, team_id):
        ax_label = fig.add_subplot(gs[row_idx, :])
        ax_label.axis('off')
        color = (_bgr_to_rgb_float(team_colors[team_id])
                 if team_id in team_colors else fallback_color[team_id])
        n = len(team_players[team_id])
        ax_label.text(0.5, 0.5,
                      f'TEAM {team_id}  ·  {n} players',
                      ha='center', va='center',
                      fontsize=14, fontweight='bold',
                      color=color,
                      transform=ax_label.transAxes)
        ax_label.set_facecolor('#111827')
        # Divider line
        ax_label.axhline(0.1, color=color, linewidth=1.5, alpha=0.4,
                         xmin=0.05, xmax=0.95)

    def _player_grid(start_row, team_id):
        players = team_players[team_id]
        cmap = cmap_per_team[team_id]
        color = (_bgr_to_rgb_float(team_colors[team_id])
                 if team_id in team_colors else fallback_color[team_id])

        for i, player_id in enumerate(players):
            row = start_row + i // COLS
            col = i % COLS
            ax = fig.add_subplot(gs[row, col])
            _draw_mini_pitch(ax)
            _kde_heatmap(ax, player_positions[player_id], cmap)

            n_frames = len(player_positions[player_id])
            ax.set_title(f'#{player_id}  ({n_frames} frames)',
                         fontsize=7.5, color=color,
                         fontweight='bold', pad=3)

        # Hide unused cells in the last row
        total_cells = math.ceil(len(players) / COLS) * COLS
        for j in range(len(players), total_cells):
            row = start_row + j // COLS
            col = j % COLS
            ax_empty = fig.add_subplot(gs[row, col])
            ax_empty.axis('off')
            ax_empty.set_facecolor('#111827')

    # Team 1
    _team_label_row(0, 1)
    _player_grid(1, 1)

    # Team 2
    _team_label_row(1 + rows_t1, 2)
    _player_grid(1 + rows_t1 + 1, 2)

    total_players = len(player_positions)
    fig.suptitle(f'Individual Player Heatmaps  ·  {total_players} players tracked',
                 fontsize=16, color='white', fontweight='bold', y=1.005)

    plt.savefig(output_path, dpi=140, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualization] Individual heatmap saved → {output_path}")

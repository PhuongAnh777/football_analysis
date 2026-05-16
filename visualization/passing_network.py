import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

# Pitch dimensions matching ViewTransformer (meters)
PITCH_LENGTH = 23.32
PITCH_WIDTH = 68.0

# Maximum frames between two possession events to still count as a pass
MAX_PASS_GAP_FRAMES = 50


def _bgr_to_rgb_float(bgr_color):
    b, g, r = bgr_color
    return (r / 255.0, g / 255.0, b / 255.0)


def _draw_pitch(ax, length=PITCH_LENGTH, width=PITCH_WIDTH):
    ax.set_facecolor('#2d5a1b')

    stripe_w = length / 10
    for i in range(10):
        color = '#2d5a1b' if i % 2 == 0 else '#347322'
        ax.add_patch(patches.Rectangle((i * stripe_w, 0), stripe_w, width,
                                       facecolor=color, zorder=0))

    ax.add_patch(patches.Rectangle((0, 0), length, width,
                                   fill=False, edgecolor='white',
                                   linewidth=2, zorder=1))
    ax.axvline(x=length / 2, color='white', linewidth=1.5, alpha=0.6, zorder=1)
    ax.scatter([length / 2], [width / 2], s=20, c='white', zorder=2)

    ax.set_xlim(-0.5, length + 0.5)
    ax.set_ylim(-0.5, width + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')


def _detect_passes(tracks):
    """
    Scan the ``has_ball`` flags frame-by-frame and identify pass events.

    A pass is recorded when:
    - Ball possession transfers from player A → player B
    - Both players belong to the **same team**
    - The gap since the last possession event is ≤ MAX_PASS_GAP_FRAMES

    Returns
    -------
    list of (from_player_id, to_player_id, team_id)
    """
    passes = []
    prev_player_id = None
    prev_team = None
    frames_without_possession = 0

    for frame_players in tracks['players']:
        current_player_id = None
        current_team = None

        for player_id, info in frame_players.items():
            if info.get('has_ball', False):
                current_player_id = player_id
                current_team = info.get('team')
                break

        if current_player_id is not None:
            if (prev_player_id is not None
                    and current_player_id != prev_player_id
                    and current_team == prev_team
                    and frames_without_possession <= MAX_PASS_GAP_FRAMES):
                passes.append((prev_player_id, current_player_id, current_team))

            prev_player_id = current_player_id
            prev_team = current_team
            frames_without_possession = 0
        else:
            frames_without_possession += 1

    return passes


def _player_avg_positions(tracks):
    """Return ``{player_id: (avg_x, avg_y)}`` using all valid transformed positions."""
    sums = defaultdict(lambda: [0.0, 0.0, 0])
    for frame_players in tracks['players']:
        for player_id, info in frame_players.items():
            pos = info.get('position_transformed')
            if pos is not None:
                sums[player_id][0] += pos[0]
                sums[player_id][1] += pos[1]
                sums[player_id][2] += 1

    return {pid: (v[0] / v[2], v[1] / v[2])
            for pid, v in sums.items() if v[2] > 0}


def _player_teams(tracks):
    """Return ``{player_id: team_id}``."""
    result = {}
    for frame_players in tracks['players']:
        for player_id, info in frame_players.items():
            team = info.get('team')
            if team is not None:
                result[player_id] = team
    return result


def generate_passing_network(tracks, team_colors,
                              output_path='output_videos/passing_network.png'):
    """
    Detect passes from the tracking data and draw a passing network for each
    team on a pitch diagram.

    Node size  ∝  number of times that player received or made a pass.
    Edge width ∝  number of passes between that pair of players.

    Parameters
    ----------
    tracks : dict
        The full ``tracks`` dict from the main pipeline.
    team_colors : dict
        ``{team_id: (B, G, R)}`` as produced by ``TeamAssigner.team_colors``.
    output_path : str
        Where to save the resulting PNG.
    """
    passes = _detect_passes(tracks)
    avg_positions = _player_avg_positions(tracks)
    player_team_map = _player_teams(tracks)

    # Directed pass counts: (from, to, team) → count
    pass_counts: dict = defaultdict(int)
    # Undirected involvement: player → total passes (sent + received)
    involvement: dict = defaultdict(int)

    for from_p, to_p, team in passes:
        pass_counts[(from_p, to_p, team)] += 1
        involvement[from_p] += 1
        involvement[to_p] += 1

    # Group players by team
    team_players: dict = defaultdict(set)
    for pid, tid in player_team_map.items():
        team_players[tid].add(pid)

    node_fallback = {1: '#e74c3c', 2: '#3498db'}
    title_fallback = {1: (0.9, 0.3, 0.3), 2: (0.3, 0.6, 0.95)}

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor('#111827')

    for idx, team_id in enumerate([1, 2]):
        ax = axes[idx]
        _draw_pitch(ax)

        players_in_team = team_players.get(team_id, set())

        # ── edges ──────────────────────────────────────────────────────────
        # Aggregate directed passes into undirected for display width
        edge_counts: dict = defaultdict(int)
        for (from_p, to_p, team), cnt in pass_counts.items():
            if team == team_id:
                key = (min(from_p, to_p), max(from_p, to_p))
                edge_counts[key] += cnt

        max_edge = max(edge_counts.values(), default=1) or 1

        for (p1, p2), cnt in edge_counts.items():
            if p1 not in avg_positions or p2 not in avg_positions:
                continue
            x1, y1 = avg_positions[p1]
            x2, y2 = avg_positions[p2]
            lw = 0.8 + 6.0 * (cnt / max_edge)
            alpha = 0.25 + 0.70 * (cnt / max_edge)
            ax.plot([x1, x2], [y1, y2],
                    color='white', linewidth=lw, alpha=alpha, zorder=2,
                    solid_capstyle='round')
            # Pass count badge at midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, str(cnt), ha='center', va='center',
                    fontsize=7, color='#111827', fontweight='bold', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              alpha=0.75, edgecolor='none'))

        # ── nodes ──────────────────────────────────────────────────────────
        node_color = (_bgr_to_rgb_float(team_colors[team_id])
                      if team_id in team_colors else node_fallback[team_id])
        max_inv = max((involvement[p] for p in players_in_team), default=1) or 1

        for player_id in players_in_team:
            if player_id not in avg_positions:
                continue
            x, y = avg_positions[player_id]
            inv = involvement.get(player_id, 0)
            size = 180 + 700 * (inv / max_inv)
            ax.scatter(x, y, s=size, c=[node_color], zorder=4,
                       edgecolors='white', linewidths=1.8)
            ax.text(x, y, str(player_id),
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold', zorder=5)

        # ── title ──────────────────────────────────────────────────────────
        title_color = (_bgr_to_rgb_float(team_colors[team_id])
                       if team_id in team_colors else title_fallback[team_id])
        total_passes = sum(v for (_, _, t), v in pass_counts.items()
                           if t == team_id)
        ax.set_title(f'Team {team_id}  ·  {total_passes} passes detected',
                     fontsize=15, color=title_color,
                     fontweight='bold', pad=12)

    # ── legend note ────────────────────────────────────────────────────────
    fig.text(0.5, -0.01,
             'Node size = pass involvement  |  Edge width = pass frequency',
             ha='center', fontsize=11, color='#9ca3af')

    fig.suptitle('Passing Network', fontsize=22,
                 color='white', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualization] Passing network saved → {output_path}")

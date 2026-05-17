"""
Formation visualizer: renders a partial-pitch diagram with player positions
colour-coded by tactical line and annotated with the detected formation string.

Coordinate system mirrors ViewTransformer output:
  x ∈ [0, 23.32] m  — along the pitch (attacking direction)
  y ∈ [0, 68]    m  — across the pitch width
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from .tactical_metrics import (
    _get_team_positions,
    _infer_attack_direction,
    _cluster_players_by_line,
    detect_formation,
    _PITCH_X_MAX,
    _PITCH_Y_MAX,
)

# ── colour palette (one per cluster/line, distinct and pitch-friendly) ─────────
_LINE_COLOURS = ["#FFD700", "#FF6B35", "#00BFFF", "#7CFC00", "#FF69B4"]

# ── pitch drawing constants ────────────────────────────────────────────────────
_PITCH_BG      = "#2d8a3e"   # grass green
_PITCH_STRIPE  = "#2c8040"   # slightly darker alternating stripe
_LINE_COLOR    = "white"
_PLAYER_RADIUS = 1.0         # metres — dot size on pitch
_FONT_COLOR    = "white"


def _draw_pitch(ax: plt.Axes) -> None:
    """
    Draw the visible pitch section (x: 0→_PITCH_X_MAX, y: 0→_PITCH_Y_MAX)
    with standard white line markings.
    """
    # Background
    ax.set_facecolor(_PITCH_BG)

    # Alternating vertical stripes for grass texture
    stripe_w = _PITCH_X_MAX / 6
    for i in range(6):
        if i % 2 == 1:
            ax.add_patch(patches.Rectangle(
                (i * stripe_w, 0), stripe_w, _PITCH_Y_MAX,
                facecolor=_PITCH_STRIPE, zorder=0
            ))

    # Pitch outline
    ax.add_patch(patches.Rectangle(
        (0, 0), _PITCH_X_MAX, _PITCH_Y_MAX,
        linewidth=2, edgecolor=_LINE_COLOR, facecolor="none", zorder=1
    ))

    # Centre line (if visible — at x ≈ 52.5 m on a full pitch, likely off-view)
    # Draw a "partial" dividing line at the midpoint of this view segment
    mid_x = _PITCH_X_MAX / 2
    ax.add_line(Line2D([mid_x, mid_x], [0, _PITCH_Y_MAX],
                       color=_LINE_COLOR, linewidth=1.5, linestyle="--",
                       alpha=0.6, zorder=1))

    # Goal line markers (left and right edges)
    for x_pos in (0.0, _PITCH_X_MAX):
        ax.add_line(Line2D([x_pos, x_pos], [_PITCH_Y_MAX * 0.25, _PITCH_Y_MAX * 0.75],
                           color=_LINE_COLOR, linewidth=3, zorder=1))

    # Touch-line tick marks every 10 m along width
    for y in range(0, int(_PITCH_Y_MAX) + 1, 10):
        ax.add_line(Line2D([0, 0.4], [y, y],
                           color=_LINE_COLOR, linewidth=1, zorder=1))
        ax.add_line(Line2D([_PITCH_X_MAX - 0.4, _PITCH_X_MAX], [y, y],
                           color=_LINE_COLOR, linewidth=1, zorder=1))


def plot_formation(
    tracks: dict,
    team_id,
    frame_idx: int,
    output_path: str,
    *,
    title_prefix: str = "",
) -> None:
    """
    Save a tactical formation diagram to *output_path* (PNG/SVG/PDF).

    The diagram shows:
    • The visible pitch section as a green rectangle with white markings
    • Player positions as coloured circles, one colour per tactical line
    • Formation string as the plot title
    • A legend identifying each tactical line

    Parameters
    ----------
    tracks      : full pipeline tracks dict
    team_id     : team identifier to visualise
    frame_idx   : which frame to draw
    output_path : file path for the saved image (extension determines format)
    title_prefix: optional prefix added before "Team X — Formation: Y-Z-W"
    """
    positions  = _get_team_positions(tracks, team_id, frame_idx)
    formation  = detect_formation(tracks, team_id, frame_idx)
    attack_dir = _infer_attack_direction(tracks, team_id)

    # ── figure & axes ─────────────────────────────────────────────────────────
    # x-axis = pitch length (0→23.32 m), y-axis = pitch width (0→68 m)
    fig_w = 7
    fig_h = fig_w * (_PITCH_Y_MAX / _PITCH_X_MAX) * 0.55   # aspect ~portrait-ish
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#1a3a2a")

    _draw_pitch(ax)

    legend_handles = []

    if len(positions) >= 2:
        pts      = np.array(positions)
        x_vals   = pts[:, 0] * attack_dir
        labels, n_clusters, centers = _cluster_players_by_line(x_vals)

        line_names = []
        # Determine line names: first cluster may be GK
        for i in range(n_clusters):
            count = int(np.sum(labels == i))
            if i == 0 and count == 1:
                line_names.append("GK")
            else:
                idx = i - (1 if line_names and line_names[0] == "GK" else 0)
                role = ["Defence", "Midfield", "Attack", "Wide"]
                line_names.append(role[idx] if idx < len(role) else f"Line {i+1}")

        for cluster_idx in range(n_clusters):
            mask   = labels == cluster_idx
            colour = _LINE_COLOURS[cluster_idx % len(_LINE_COLOURS)]
            name   = line_names[cluster_idx]

            ax.scatter(
                pts[mask, 0], pts[mask, 1],
                s=220,
                color=colour,
                edgecolors=_LINE_COLOR,
                linewidths=1.5,
                zorder=4,
            )
            # Player number label (cluster-local index)
            for j, (px, py) in enumerate(pts[mask], start=1):
                ax.text(
                    px, py, str(j),
                    ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color="#1a1a1a", zorder=5,
                )

            legend_handles.append(
                Line2D([0], [0],
                       marker="o", color="none",
                       markerfacecolor=colour, markeredgecolor=_LINE_COLOR,
                       markersize=10, label=name)
            )

    elif len(positions) == 1:
        # Single player — draw without clustering
        pt = np.array(positions[0])
        ax.scatter(pt[0], pt[1], s=220, color=_LINE_COLOURS[0],
                   edgecolors=_LINE_COLOR, linewidths=1.5, zorder=4)
    else:
        ax.text(
            _PITCH_X_MAX / 2, _PITCH_Y_MAX / 2,
            "No player data",
            ha="center", va="center", fontsize=12,
            color=_FONT_COLOR, zorder=5,
        )

    # Attack direction arrow
    arrow_y = _PITCH_Y_MAX + 3.5
    arrow_dx = 3.0 * attack_dir
    arrow_x0 = (_PITCH_X_MAX / 2) - arrow_dx / 2
    ax.annotate(
        "Attack",
        xy=(arrow_x0 + arrow_dx, arrow_y),
        xytext=(arrow_x0, arrow_y),
        arrowprops=dict(arrowstyle="->", color=_FONT_COLOR, lw=1.5),
        color=_FONT_COLOR, fontsize=8, ha="center", va="center",
        annotation_clip=False,
    )

    # ── axes formatting ────────────────────────────────────────────────────────
    ax.set_xlim(-1.5, _PITCH_X_MAX + 1.5)
    ax.set_ylim(-6, _PITCH_Y_MAX + 2)
    ax.set_xlabel("Pitch length (m)", color=_FONT_COLOR, fontsize=9)
    ax.set_ylabel("Pitch width (m)",  color=_FONT_COLOR, fontsize=9)
    ax.tick_params(colors=_FONT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_FONT_COLOR)

    title = (
        f"{title_prefix}Team {team_id}  •  Frame {frame_idx}"
        f"  •  Formation: {formation}"
    )
    ax.set_title(title, color=_FONT_COLOR, fontsize=11, pad=8)

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.4,
            facecolor="#1a3a2a",
            edgecolor=_LINE_COLOR,
            labelcolor=_FONT_COLOR,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

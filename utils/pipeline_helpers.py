"""Shared helpers used by main.py and the API pipeline runner."""

from __future__ import annotations

from player_ball_assigner import PlayerBallAssigner


def assign_ball_to_tracks(tracks: dict) -> list[int]:
    """Assign ball possession per frame; return team_ball_control list."""
    player_assigner = PlayerBallAssigner()
    team_ball_control: list[int] = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_frame = tracks["ball"][frame_num]
        ball_entry = ball_frame.get(1) if ball_frame else None

        if ball_entry is None or "bbox" not in ball_entry:
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else 0
            )
            continue

        assigned = player_assigner.assign_ball_to_player(
            player_track, ball_entry["bbox"]
        )
        if assigned != -1:
            tracks["players"][frame_num][assigned]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned]["team"]
            )
        else:
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else 0
            )

    return team_ball_control


def extract_passing_events(tracks: dict) -> list[dict]:
    """Detect pass events from has_ball flags."""
    events: list[dict] = []
    prev_carrier: dict[int, int | None] = {1: None, 2: None}

    for frame_idx, frame_data in enumerate(tracks["players"]):
        for track_id, data in frame_data.items():
            if not data.get("has_ball"):
                continue
            team = data.get("team")
            if team not in (1, 2):
                continue
            prev = prev_carrier[team]
            if prev is not None and prev != track_id:
                passer_pos = tracks["players"][frame_idx].get(prev, {}).get(
                    "position_transformed"
                )
                receiver_pos = data.get("position_transformed")
                events.append(
                    {
                        "frame": frame_idx,
                        "team": team,
                        "passer_id": prev,
                        "receiver_id": track_id,
                        "passer_pos": passer_pos,
                        "receiver_pos": receiver_pos,
                        "success": True,
                    }
                )
            prev_carrier[team] = track_id

    return events


def extract_failed_pass_events(
    tracks: dict,
    team_ball_control: list[int],
) -> list[dict]:
    """Detect failed passes: possession leaves the team from the last carrier.

  A pass is **successful** when a teammate receives the ball (see
  ``extract_passing_events``).  A pass is **failed** when the ball carrier's
  team loses possession to the opponent without a same-team reception.
    """
    events: list[dict] = []
    n_frames = min(len(tracks.get("players", [])), len(team_ball_control))

    for i in range(1, n_frames):
        prev_team = team_ball_control[i - 1]
        curr_team = team_ball_control[i]
        if prev_team not in (1, 2) or curr_team in (prev_team, 0):
            continue

        frame = tracks["players"][i - 1]
        for tid, data in frame.items():
            if data.get("has_ball") and data.get("team") == prev_team:
                events.append({
                    "frame":     i - 1,
                    "team":      prev_team,
                    "passer_id": int(tid),
                    "success":   False,
                })
                break

    return events


# Midfield line along pos[0] when ViewTransformer uses full 105 m pitch offsets.
_PITCH_MID_M = 105.0 / 2.0
_MIN_CTRL_FRAMES = 3


def extract_defensive_events(
    tracks: dict,
    team_ball_control: list[int],
    *,
    pitch_mid: float = _PITCH_MID_M,
) -> list[dict]:
    """Infer defensive actions from ball-recovery events for PPDA.

    Each sustained possession change where *team* gains the ball in the
    opponent's half (x >= *pitch_mid*) is recorded as an ``interception``.
    """
    events: list[dict] = []
    ctrl = team_ball_control
    n_pframes = len(tracks.get("players", []))

    for team_idx in (1, 2):
        i = 1
        while i < len(ctrl):
            if ctrl[i] == team_idx and ctrl[i - 1] != team_idx:
                j = i
                while j < len(ctrl) and ctrl[j] == team_idx:
                    j += 1
                if j - i < _MIN_CTRL_FRAMES:
                    i = j
                    continue

                pos = None
                for fi in range(i, min(i + 5, n_pframes)):
                    for info in tracks["players"][fi].values():
                        if (
                            info.get("team") == team_idx
                            and info.get("has_ball")
                            and info.get("position_transformed") is not None
                        ):
                            pos = info["position_transformed"]
                            break
                    if pos is not None:
                        break

                if pos is not None and float(pos[0]) >= pitch_mid:
                    events.append({
                        "frame": i,
                        "team":  team_idx,
                        "type":  "interception",
                        "x":     float(pos[0]),
                        "y":     float(pos[1]),
                    })
                i = j
            else:
                i += 1

    return events

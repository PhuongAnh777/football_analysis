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
                    }
                )
            prev_carrier[team] = track_id

    return events

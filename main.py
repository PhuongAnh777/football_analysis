from utils import read_video, save_video
from trackers import Tracker, merge_player_tracks
from team_assigner import TeamAssigner
import cv2
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from visualization import generate_heatmap, generate_passing_network
from tactical_analyzer import (
    analyze_all_frames, plot_formation,
    evaluate_tactics, generate_report,
)

def main():
    # Read video
    video_frames = read_video("input_videos/input_video.mp4")

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks( video_frames, 
                                        read_from_stub = True, 
                                        stub_path='stubs/track_stubs.pkl')

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                               read_from_stub = True,
                                                                               stub_path='stubs/camera_movement_stub.pkl')
    # Add adjusted positions to tracks
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Transform tracks
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Draw transformed tracks
    view_transformer = ViewTransformer()

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Interpolate ball position
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    # Initialize Team Assigner
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks["players"][0])
    
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], 
                                                track['bbox'], 
                                                player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # Merge fragmented player track IDs caused by camera cuts/pans
    # Must run after team assignment (merger filters by team) and after
    # position_transformed is populated (used for spatial linking).
    ids_before = len({tid for frame in tracks["players"] for tid in frame})
    try:
        tracks = merge_player_tracks(tracks)
        ids_after = len({tid for frame in tracks["players"] for tid in frame})
        print(f"[TrackMerger] Unique player IDs: {ids_before} → {ids_after}")
    except Exception as e:
        print(f"[TrackMerger] WARNING: merge failed ({e}), continuing with original tracks")

    # Assign ball to player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    # Initialize Player Ball Assigner
    # Save cropped image of a player
    for track_id, player in tracks["players"][0].items():
        bbox = player['bbox']
        frame = video_frames[0]

        # Crop bbox from frame
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Save cropped image
        cv2.imwrite(f"output_videos/cropped_image.jpg", cropped_image)

        break
    
    team_ball_control = np.array(team_ball_control)

    # Draw output video (in-place on video_frames — single copy in memory)
    tracker.draw_annotations(video_frames, tracks, team_ball_control)
    camera_movement_estimator.draw_camera_movement(video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(video_frames, tracks)

    save_video(video_frames, "output_videos/output_video.avi")

    # ── Tactical analysis ──────────────────────────────────────────────────
    # Requires: team, position_transformed, speed, has_ball — all populated above.
    try:
        tactical_report = analyze_all_frames(tracks, sample_every=30)

        # Per-frame console log
        for team_id, samples in tactical_report.items():
            for s in samples:
                compact = f"{s['compact']:.1f}m" if s['compact'] is not None else "N/A"
                speed   = f"{s['avg_speed']:.1f}km/h" if s['avg_speed'] is not None else "N/A"
                print(
                    f"[Tactics] Team {team_id}  frame {s['frame']:4d}  "
                    f"formation={s['formation']:8s}  compact={compact:>8s}  "
                    f"pressing={s['pressing']:.2f}  speed={speed}"
                )

        # Formation diagram for each team at the mid-point frame
        mid_frame = len(tracks["players"]) // 2
        for team_id in tactical_report:
            plot_formation(
                tracks, team_id, mid_frame,
                output_path=f"output_videos/formation_team{team_id}_frame{mid_frame}.png",
            )

        # ── Possession % from team_ball_control ───────────────────────────
        from collections import Counter as _Counter
        ball_counts    = _Counter(team_ball_control.tolist())
        total_frames   = max(len(team_ball_control), 1)
        possession_pct = {t: round(c / total_frames * 100, 1)
                          for t, c in ball_counts.items()}

        # ── Total distance per team (sum of each player's last distance) ──
        # The speed estimator stores cumulative distance in each frame;
        # we take the last non-None value per player then sum by team.
        _latest_dist: dict = {}
        for frame in tracks["players"]:
            for tid, info in frame.items():
                dist = info.get("distance")
                if dist is not None:
                    _latest_dist[tid] = (info.get("team"), dist)

        total_distance: dict = {}
        for _team, _d in _latest_dist.values():
            if _team is not None:
                total_distance[_team] = total_distance.get(_team, 0.0) + _d

        # ── Rule-based evaluation ─────────────────────────────────────────
        evaluation = evaluate_tactics(tactical_report, possession_pct, total_distance)
        print("\n[RuleEngine] Tactical evaluation:")
        for team_id, summary in evaluation.items():
            if team_id == "match_events":
                continue
            print(f"  Team {team_id}: {summary}")
        for event in evaluation.get("match_events", []):
            print(f"  [event] {event}")

        # ── LLM report (optional — controlled by LLM_PROVIDER env var) ───
        # Set LLM_PROVIDER=ollama (requires `ollama serve`) or
        # LLM_PROVIDER=openai (requires OPENAI_API_KEY) to enable.
        # Leave LLM_PROVIDER unset to skip gracefully.
        import os as _os
        if _os.getenv("LLM_PROVIDER"):
            try:
                report = generate_report(evaluation)
                report_path = "output_videos/tactical_report.txt"
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"\n[LLMReporter] Report saved → {report_path}")
                print(report)
            except Exception as llm_err:
                print(f"[LLMReporter] WARNING: LLM call failed ({llm_err})")

    except Exception as e:
        print(f"[TacticalAnalyzer] WARNING: analysis failed ({e}), skipping")

    # ── Visualization ──────────────────────────────────────────────────────
    generate_heatmap(
        tracks,
        team_assigner.team_colors,
        output_path='output_videos/heatmap.png'
    )

    generate_passing_network(
        tracks,
        team_assigner.team_colors,
        output_path='output_videos/passing_network.png'
    )

if __name__ == "__main__":
    main()
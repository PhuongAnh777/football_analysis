from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
import cv2
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from visualization import generate_heatmap, generate_passing_network

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
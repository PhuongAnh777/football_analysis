import json

import os

from utils import read_video, save_video
from utils.pipeline_helpers import (
    assign_ball_to_tracks,
    extract_defensive_events,
    extract_passing_events,
)
from utils.stub_io import load_track_stub

from trackers import Tracker, merge_player_tracks

from team_assigner import TeamAssigner

from utils.scoreboard_reader import detect_scoreboard_stripe_colors

import cv2

import numpy as np

from camera_movement_estimator import CameraMovementEstimator

from view_transformer import ViewTransformer

from speed_and_distance_estimator import SpeedAndDistance_Estimator

from visualization import generate_heatmap, generate_passing_network

from tactical_analyzer import (

    TacticalAnalyzer,

    ThresholdEngine,

    ReportBuilder,

    TacticalNarrator,

)



LLM_API_KEY  = os.getenv("OPENAI_API_KEY", "")

LLM_MODEL    = os.getenv("LLM_MODEL",      "gpt-4o")

LLM_BASE_URL = os.getenv("LLM_BASE_URL",   "https://api.openai.com/v1")





class _NumpyEncoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, np.integer):

            return int(obj)

        if isinstance(obj, np.floating):

            return float(obj)

        if isinstance(obj, np.ndarray):

            return obj.tolist()

        return super().default(obj)





def main():

    video_frames, fps = read_video("input_videos/input_video_45s.mp4")

    fps_int = max(1, round(fps))



    tracker = Tracker('models/best.pt')

    stub_path = 'stubs/track_stubs.pkl'
    if os.path.exists(stub_path):
        print(f"[main] Loading Colab tracking stub: {stub_path}")
        tracks, stub_fps, enriched = load_track_stub(stub_path)
        if stub_fps is not None:
            fps = stub_fps
            fps_int = max(1, round(fps))
        if not enriched:
            tracker.add_appearance_to_tracks(tracks, video_frames)
            tracker.add_position_to_tracks(tracks)
    else:
        tracks = tracker.get_object_tracks(
            video_frames,
            read_from_stub=False,
        )
        tracker.add_appearance_to_tracks(tracks, video_frames)
        tracker.add_position_to_tracks(tracks)



    camera_movement_estimator = CameraMovementEstimator(video_frames[0])

    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(

        video_frames,

        read_from_stub=True,

        stub_path='stubs/camera_movement_stub.pkl',

    )

    camera_movement_estimator.add_adjust_positions_to_tracks(

        tracks, camera_movement_per_frame

    )



    frame_w = video_frames[0].shape[1]
    frame_h = video_frames[0].shape[0]
    view_transformer = ViewTransformer(frame_size=(frame_w, frame_h))

    cumulative_cam = CameraMovementEstimator.cumulative(camera_movement_per_frame)
    pitch_offsets  = view_transformer.compute_pitch_offsets(
        cumulative_cam, pitch_x_start=0.0
    )
    view_transformer.add_transformed_position_to_tracks(tracks, pitch_offsets=pitch_offsets)

    # ── In kích thước sân đo được ra console ─────────────────────────────────
    _all_x, _all_y = [], []
    for _frame in tracks["players"]:
        for _info in _frame.values():
            _pos = _info.get("position_transformed")
            if _pos is not None:
                _all_x.append(_pos[0])
                _all_y.append(_pos[1])
    if _all_x:
        _x_span = max(_all_x) - min(_all_x)
        _y_span = max(_all_y) - min(_all_y)
        print(
            f"[Pitch] Kích thước sân đo được: "
            f"dài = {_x_span:.1f} m  ({min(_all_x):.1f}→{max(_all_x):.1f} m)  |  "
            f"rộng = {_y_span:.1f} m  ({min(_all_y):.1f}→{max(_all_y):.1f} m)"
        )
        print(
            f"[Pitch] m_per_px = {view_transformer.pan_scale_mpp():.4f}  |  "
            f"offset span = {max(pitch_offsets)-min(pitch_offsets):.1f} m"
        )
    # ─────────────────────────────────────────────────────────────────────────

    speed_and_distance_estimator = SpeedAndDistance_Estimator(fps=fps_int)

    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)



    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])



    team_assigner = TeamAssigner()
    calib_data: list = []
    for _fi, _pt in enumerate(tracks["players"]):
        if len(_pt) >= TeamAssigner._MIN_PLAYERS:
            calib_data.append((video_frames[_fi], _pt))
        if len(calib_data) >= TeamAssigner._CALIB_FRAMES:
            break
    if not calib_data:
        _fb = next((i for i, p in enumerate(tracks["players"]) if len(p) >= 2), 0)
        calib_data = [(video_frames[_fb], tracks["players"][_fb])]
    team_assigner.assign_team_color(calib_data)

    stripe_left, stripe_right = detect_scoreboard_stripe_colors(
        video_frames,
        api_key=LLM_API_KEY or None,
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
    )
    if stripe_left is not None and stripe_right is not None:
        team_assigner.align_to_scoreboard(stripe_left, stripe_right)



    for frame_num, player_track in enumerate(tracks["players"]):

        for player_id, track in player_track.items():

            team = team_assigner.get_player_team(

                video_frames[frame_num], track['bbox'], player_id

            )

            tracks["players"][frame_num][player_id]["team"] = team

            tracks["players"][frame_num][player_id]["team_color"] = (

                team_assigner.team_colors[team]

            )



    tracks = merge_player_tracks(tracks)

    team_ball_control = assign_ball_to_tracks(tracks)



    for track_id, player in tracks["players"][0].items():

        bbox = player['bbox']

        frame = video_frames[0]

        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        cv2.imwrite("output_videos/cropped_image.jpg", cropped_image)

        break



    os.makedirs("output_videos", exist_ok=True)

    passing_events = extract_passing_events(tracks)
    defensive_events = extract_defensive_events(tracks, team_ball_control)

    print(f"      Detected {len(passing_events)} passing events "
          f"(team 1: {sum(1 for e in passing_events if e['team']==1)}, "
          f"team 2: {sum(1 for e in passing_events if e['team']==2)})")
    print(f"      Inferred {len(defensive_events)} defensive events for PPDA")

    print("[1/4] Running TacticalAnalyzer...")
    analyzer = TacticalAnalyzer(fps=fps_int, window_sec=30, R_pressing=8.0)
    tactical_report = analyzer.analyze(
        tracks,
        team_ball_control,
        passing_events=passing_events,
        defensive_events=defensive_events,
    )

    with open("output_videos/tactical_report.json", "w", encoding="utf-8") as f:

        json.dump(tactical_report, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)



    print("[2/4] Running ThresholdEngine...")

    engine = ThresholdEngine(fps=fps_int, R_pressing=8.0)

    scored_report = engine.compute(tactical_report, tracks)

    with open("output_videos/scored_report.json", "w", encoding="utf-8") as f:

        json.dump(scored_report, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)



    print("[3/4] Running ReportBuilder...")

    builder = ReportBuilder(window_frames=int(30 * fps_int))

    match_report = builder.build(

        scored_report=scored_report,

        tactical_report=tactical_report,

        total_frames=len(tracks["players"]),

    )

    with open("output_videos/match_report.json", "w", encoding="utf-8") as f:

        json.dump(match_report, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)



    if LLM_API_KEY:

        print("[4/4] Running TacticalNarrator (LLM)...")

        narrator = TacticalNarrator(

            api_key=LLM_API_KEY,

            model=LLM_MODEL,

            base_url=LLM_BASE_URL,

        )

        try:

            llm_analysis = narrator.analyze(match_report)

            with open("output_videos/llm_analysis.json", "w", encoding="utf-8") as f:

                json.dump(llm_analysis, f, ensure_ascii=False, indent=2)

        except Exception as exc:

            print(f"      [WARNING] LLM analysis failed: {exc}")

    else:

        print("[4/4] Skipping TacticalNarrator (OPENAI_API_KEY not set).")



    team_ball_control = np.array(team_ball_control)

    tracker.draw_annotations(video_frames, tracks, team_ball_control)

    speed_and_distance_estimator.draw_speed_and_distance(video_frames, tracks)

    save_video(video_frames, "output_videos/output_video.avi", fps=fps)



    generate_heatmap(

        tracks, team_assigner.team_colors, output_path='output_videos/heatmap.png'

    )

    generate_passing_network(

        tracks, team_assigner.team_colors, output_path='output_videos/passing_network.png'

    )





if __name__ == "__main__":

    main()


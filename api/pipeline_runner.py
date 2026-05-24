"""
pipeline_runner.py
==================
Wraps the Football Analysis pipeline (main.py logic) so it can be
executed from the FastAPI background task, with per-step progress
reporting written directly into the in-memory job store.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import traceback as tb
from typing import Callable, Optional

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import read_video, save_video
from utils.pipeline_helpers import assign_ball_to_tracks, extract_passing_events
from utils.stub_io import load_track_stub
from trackers import Tracker, merge_player_tracks
from team_assigner import TeamAssigner
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

from api.job_store import JobState
from api.result_adapter import adapt_api_result, build_streamlit_analysis_result

_TOTAL_STEPS = 8
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output_videos")
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
_DEFAULT_TRACK_STUB = os.path.join(_PROJECT_ROOT, "stubs", "track_stubs.pkl")

_PIPELINE_STEPS = [
    (1, "reading",  "Đang đọc video..."),
    (2, "tracking", "Đang tracking cầu thủ..."),
    (3, "camera",   "Đang phân tích camera movement..."),
    (4, "teams",    "Đang gán đội hình..."),
    (5, "speed",    "Đang tính tốc độ & khoảng cách..."),
    (6, "tactical", "Đang phân tích chiến thuật..."),
    (7, "report",   "Đang tạo báo cáo..."),
    (8, "render",   "Đang render output..."),
]


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _encode_image(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def _filter_tracks_by_team(tracks: dict, team_id: int) -> dict:
    filtered_players = [
        {pid: info for pid, info in frame.items() if info.get("team") == team_id}
        for frame in tracks["players"]
    ]
    return {**tracks, "players": filtered_players}


def _dump_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, cls=_NumpyEncoder)


def _convert_to_mp4(avi_path: str) -> str:
    mp4_path = avi_path.replace(".avi", ".mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", avi_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",
                "-an",
                mp4_path,
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )
        return mp4_path
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return avi_path


def _should_use_track_stub(track_stub_path: str | None) -> str | None:
    """Return stub path when Colab stub exists and tracking is not forced."""
    force = os.getenv("FORCE_TRACKING", "").lower() in ("1", "true", "yes")
    if force:
        return None
    path = track_stub_path or os.getenv("TRACK_STUB_PATH", _DEFAULT_TRACK_STUB)
    return path if path and os.path.exists(path) else None


def execute_pipeline(
    video_path: str,
    *,
    read_from_stub: bool = False,
    track_stub_path: str | None = None,
    on_step: Optional[Callable[[int, str, str], None]] = None,
) -> dict:
    """Run the full CV + tactical pipeline and return raw + adapted outputs."""

    def _notify(step_n: int, step_key: str, label: str) -> None:
        if on_step:
            on_step(step_n, step_key, label)

    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    _notify(1, "reading", _PIPELINE_STEPS[0][2])
    video_frames, fps = read_video(video_path)
    fps_int = max(1, round(fps))

    _notify(2, "tracking", _PIPELINE_STEPS[1][2])
    tracker = Tracker(os.path.join(_MODELS_DIR, "best.pt"))

    colab_stub = _should_use_track_stub(track_stub_path)
    if colab_stub:
        print(f"[pipeline] Loading Colab tracking stub: {colab_stub}", flush=True)
        tracks, stub_fps, enriched = load_track_stub(colab_stub)
        if stub_fps is not None:
            fps = stub_fps
            fps_int = max(1, round(fps))
        if not enriched:
            tracker.add_appearance_to_tracks(tracks, video_frames)
            tracker.add_position_to_tracks(tracks)
    else:
        tracks = tracker.get_object_tracks(
            video_frames,
            read_from_stub=read_from_stub,
            stub_path=_DEFAULT_TRACK_STUB if read_from_stub else None,
        )
        tracker.add_appearance_to_tracks(tracks, video_frames)
        tracker.add_position_to_tracks(tracks)

    _notify(3, "camera", _PIPELINE_STEPS[2][2])
    cam_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = cam_estimator.get_camera_movement(
        video_frames, read_from_stub=read_from_stub
    )
    cam_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    _notify(4, "teams", _PIPELINE_STEPS[3][2])
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    first_frame = next(
        (i for i, p in enumerate(tracks["players"]) if len(p) >= 2), 0
    )
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(
        video_frames[first_frame], tracks["players"][first_frame]
    )

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

    tracks = merge_player_tracks(tracks)
    team_ball_control = assign_ball_to_tracks(tracks)

    _notify(5, "speed", _PIPELINE_STEPS[4][2])
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)
    passing_events = extract_passing_events(tracks)

    _notify(6, "tactical", _PIPELINE_STEPS[5][2])
    analyzer = TacticalAnalyzer(fps=fps_int, window_sec=30, R_pressing=8.0)
    tactical_report = analyzer.analyze(
        tracks, team_ball_control, passing_events=passing_events
    )

    engine = ThresholdEngine(fps=fps_int, R_pressing=8.0)
    scored_report = engine.compute(tactical_report, tracks)

    _dump_json(tactical_report, os.path.join(_OUTPUT_DIR, "tactical_report.json"))
    _dump_json(scored_report, os.path.join(_OUTPUT_DIR, "scored_report.json"))

    _notify(7, "report", _PIPELINE_STEPS[6][2])
    builder = ReportBuilder(window_frames=int(30 * fps_int))
    match_report = builder.build(
        scored_report=scored_report,
        tactical_report=tactical_report,
        total_frames=len(tracks["players"]),
    )
    _dump_json(match_report, os.path.join(_OUTPUT_DIR, "match_report.json"))

    llm_eval: dict = {}
    llm_api_key = os.getenv("OPENAI_API_KEY", "")
    if llm_api_key:
        try:
            narrator = TacticalNarrator(
                api_key=llm_api_key,
                model=os.getenv("LLM_MODEL", "gpt-4o"),
                base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            )
            llm_eval = narrator.analyze(match_report)
            _dump_json(llm_eval, os.path.join(_OUTPUT_DIR, "llm_analysis.json"))
        except Exception as exc:
            llm_eval = {"warning": f"LLM analysis failed: {exc}"}

    _notify(8, "render", _PIPELINE_STEPS[7][2])
    team_ball_control_arr = np.array(team_ball_control)
    tracker.draw_annotations(video_frames, tracks, team_ball_control_arr)
    cam_estimator.draw_camera_movement(video_frames, camera_movement_per_frame)
    speed_estimator.draw_speed_and_distance(video_frames, tracks)

    avi_path = os.path.join(_OUTPUT_DIR, "output_video.avi")
    save_video(video_frames, avi_path, fps=fps)

    video_output_path = _convert_to_mp4(avi_path)

    generate_heatmap(
        tracks,
        team_assigner.team_colors,
        output_path=os.path.join(_OUTPUT_DIR, "heatmap.png"),
    )
    generate_passing_network(
        tracks,
        team_assigner.team_colors,
        output_path=os.path.join(_OUTPUT_DIR, "passing_network.png"),
    )

    tracks_t1 = _filter_tracks_by_team(tracks, 1)
    tracks_t2 = _filter_tracks_by_team(tracks, 2)

    chart_paths = {
        "heatmap_team1": os.path.join(_OUTPUT_DIR, "heatmap_team1.png"),
        "heatmap_team2": os.path.join(_OUTPUT_DIR, "heatmap_team2.png"),
        "passing_network_team1": os.path.join(_OUTPUT_DIR, "passing_network_team1.png"),
        "passing_network_team2": os.path.join(_OUTPUT_DIR, "passing_network_team2.png"),
    }
    generate_heatmap(tracks_t1, team_assigner.team_colors, output_path=chart_paths["heatmap_team1"])
    generate_heatmap(tracks_t2, team_assigner.team_colors, output_path=chart_paths["heatmap_team2"])
    generate_passing_network(tracks_t1, team_assigner.team_colors, output_path=chart_paths["passing_network_team1"])
    generate_passing_network(tracks_t2, team_assigner.team_colors, output_path=chart_paths["passing_network_team2"])

    charts = {key: _encode_image(path) for key, path in chart_paths.items()}

    adapted = adapt_api_result(
        match_report=match_report,
        tactical_report=tactical_report,
        scored_report=scored_report,
        charts=charts,
        passing_events=passing_events,
        llm_eval=llm_eval or None,
        fps=fps,
    )

    meta = match_report.get("meta", {})
    poss = tactical_report.get("possession", {}).get("possession", {})
    dist = tactical_report.get("possession", {}).get("total_distance", {})

    return {
        "adapted": adapted,
        "match_report": match_report,
        "tactical_report": tactical_report,
        "scored_report": scored_report,
        "video_path": video_output_path,
        "streamlit": {
            "analysis_result": build_streamlit_analysis_result(tactical_report),
            "possession_pct": {
                1: float(poss.get("team_1", meta.get("possession_team_1", 0))),
                2: float(poss.get("team_2", meta.get("possession_team_2", 0))),
            },
            "total_distance": {
                1: float(dist.get("team_1", 0)),
                2: float(dist.get("team_2", 0)),
            },
            "output_video_path": video_output_path,
            "heatmap_path": os.path.join(_OUTPUT_DIR, "heatmap.png"),
            "passing_network_path": os.path.join(_OUTPUT_DIR, "passing_network.png"),
        },
    }


def run_pipeline(video_path: str, job_id: str, jobs_store: dict) -> None:
    """Execute pipeline for a background API job."""

    def _step(step_n: int, step_key: str, label: str) -> None:
        job: JobState = jobs_store[job_id]
        job.progress = (step_n - 1) / _TOTAL_STEPS
        job.current_step = label
        job.step_key = step_key

    try:
        outputs = execute_pipeline(
            video_path,
            read_from_stub=False,
            on_step=_step,
        )
        job: JobState = jobs_store[job_id]
        job.result = _sanitize(outputs["adapted"])
        job.video_path = outputs["video_path"]
        job.status = "done"
        job.progress = 1.0
        job.current_step = "Hoàn thành"
        job.step_key = "done"

    except Exception as exc:
        error_detail = f"{type(exc).__name__}: {exc}\n{tb.format_exc()}"
        job = jobs_store[job_id]
        job.status = "error"
        job.error = error_detail
        job.current_step = "Lỗi"
        job.step_key = "error"

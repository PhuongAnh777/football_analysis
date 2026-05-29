#!/usr/bin/env python3
"""
Run YOLO + BoT-SORT tracking on GPU (Colab) and save an enriched stub for local pipeline.

Usage (Colab or local GPU):
    python scripts/run_tracking.py input_videos/input_video.mp4
    python scripts/run_tracking.py input_videos/input_video.mp4 --stub stubs/track_stubs.pkl
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from trackers import Tracker
from utils import read_video
from utils.stub_io import save_track_stub


def run_tracking(
    video_path: str,
    *,
    model_path: str = "models/best.pt",
    stub_path: str = "stubs/track_stubs.pkl",
) -> str:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"[tracking] Video: {video_path}")
    print(f"[tracking] Model: {model_path}")

    video_frames, fps = read_video(video_path)
    print(f"[tracking] {len(video_frames)} frames @ {fps:.1f} fps")

    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)

    print("[tracking] ReID appearance features...")
    tracker.add_appearance_to_tracks(tracks, video_frames)
    tracker.add_position_to_tracks(tracks)

    save_track_stub(stub_path, tracks, fps, enriched=True, source_video=video_path)
    print(f"[tracking] Saved enriched stub → {stub_path}")

    del video_frames, tracks, tracker
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return stub_path


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU tracking → enriched stub for local pipeline")
    parser.add_argument("video", help="Path to input video (MP4/AVI)")
    parser.add_argument("--model", default="models/best.pt", help="YOLO weights path")
    parser.add_argument("--stub", default="stubs/track_stubs.pkl", help="Output stub path")
    args = parser.parse_args()

    os.chdir(_PROJECT_ROOT)
    run_tracking(args.video, model_path=args.model, stub_path=args.stub)


if __name__ == "__main__":
    main()

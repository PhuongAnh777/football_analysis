import os

import cv2


def blend_filled_rectangle(frame, pt1, pt2, color=(255, 255, 255), alpha=0.6):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def read_video(video_path):
    """Load all frames and return ``(frames, fps)``."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if fps <= 0 or fps > 120:
        fps = 24.0
    print(f"[read_video] {video_path}  |  {total} frames @ {fps:.1f} fps", flush=True)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) % 500 == 0:
            print(f"[read_video] {len(frames)}/{total} frames loaded...", flush=True)
    cap.release()
    print(f"[read_video] Done — {len(frames)} frames", flush=True)
    return frames, float(fps)

def save_video(output_video_frames, output_video_path, fps=24.0):
    """Ghi video; ưu tiên MP4 (mp4v) để phát được trên trình duyệt."""
    if not output_video_frames:
        raise ValueError("No frames to save")

    path = output_video_path
    if path.lower().endswith(".avi"):
        path = path[:-4] + ".mp4"

    h, w = output_video_frames[0].shape[:2]
    for fourcc_name in ("mp4v", "avc1", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        out = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
        if not out.isOpened():
            continue
        for frame in output_video_frames:
            out.write(frame)
        out.release()
        if os.path.isfile(path) and os.path.getsize(path) > 1000:
            print(f"[save_video] Wrote {path} (fourcc={fourcc_name})", flush=True)
            return path

    raise RuntimeError(f"Could not write video: {path}")

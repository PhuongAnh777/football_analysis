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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, float(fps), (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

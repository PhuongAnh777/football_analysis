import os
import shutil
import subprocess
import tempfile

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


def ffmpeg_exe() -> str | None:
    """Return path to ffmpeg (bundled via imageio-ffmpeg, or system PATH)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def is_browser_playable_mp4(path: str) -> bool:
    """True when the file looks like H.264/AAC MP4 that Chrome/Edge can play."""
    if not path.lower().endswith(".mp4") or not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as fh:
            head = fh.read(512_000)
    except OSError:
        return False
    # H.264 tracks use avc1 / h264 / avc3 in the ftyp/moov atoms.
    return any(tag in head for tag in (b"avc1", b"h264", b"avc3"))


def transcode_to_browser_mp4(src: str, dst: str | None = None) -> str | None:
    """Re-encode *src* to H.264 + yuv420p MP4 suitable for HTML5 <video>.

    Returns the output path on success, else ``None``.
    """
    ffmpeg = ffmpeg_exe()
    if not ffmpeg or not os.path.isfile(src):
        return None

    out = dst or src
    tmp = out + ".h264.tmp.mp4" if out == src else out
    try:
        subprocess.run(
            [
                ffmpeg, "-y",
                "-i", src,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                tmp,
            ],
            check=True,
            capture_output=True,
            timeout=900,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[transcode] Failed {src} → {tmp}: {exc}", flush=True)
        if os.path.isfile(tmp):
            os.remove(tmp)
        return None

    if not os.path.isfile(tmp) or os.path.getsize(tmp) < 1000:
        if os.path.isfile(tmp):
            os.remove(tmp)
        return None

    if out == src:
        os.replace(tmp, out)
    print(f"[transcode] Browser-ready MP4: {out}", flush=True)
    return out


def ensure_browser_playable(path: str) -> str:
    """Return a browser-playable MP4 path, transcoding in-place when needed."""
    if not path.lower().endswith((".mp4", ".avi")):
        return path

    if path.lower().endswith(".mp4") and is_browser_playable_mp4(path):
        return path

    # AVI or OpenCV mp4v → H.264 MP4 alongside original.
    base, _ = os.path.splitext(path)
    h264_path = base + "_h264.mp4"
    if os.path.isfile(h264_path) and is_browser_playable_mp4(h264_path):
        return h264_path

    converted = transcode_to_browser_mp4(path, h264_path)
    if converted and is_browser_playable_mp4(converted):
        return converted

    return path


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
    """Write annotated video as H.264 MP4 playable in browsers."""
    if not output_video_frames:
        raise ValueError("No frames to save")

    path = output_video_path
    if path.lower().endswith(".avi"):
        path = path[:-4] + ".mp4"

    h, w = output_video_frames[0].shape[:2]

    # Write raw frames with OpenCV, then transcode to H.264 (mp4v ≠ browser-safe).
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        raw_path = tmp.name

    written = False
    try:
        for fourcc_name in ("mp4v", "XVID", "avc1"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            out = cv2.VideoWriter(raw_path, fourcc, float(fps), (w, h))
            if not out.isOpened():
                continue
            for frame in output_video_frames:
                out.write(frame)
            out.release()
            if os.path.isfile(raw_path) and os.path.getsize(raw_path) > 1000:
                written = True
                print(f"[save_video] Raw encode {raw_path} (fourcc={fourcc_name})", flush=True)
                break

        if not written:
            raise RuntimeError(f"Could not write video: {path}")

        if is_browser_playable_mp4(raw_path):
            os.replace(raw_path, path)
            print(f"[save_video] Wrote {path} (already H.264)", flush=True)
            return path

        converted = transcode_to_browser_mp4(raw_path, path)
        if converted and is_browser_playable_mp4(converted):
            return converted

        # Last resort: keep raw file so user can still download it.
        os.replace(raw_path, path)
        print(
            f"[save_video] WARNING: {path} may not play in browser "
            f"(install imageio-ffmpeg: pip install imageio-ffmpeg)",
            flush=True,
        )
        return path
    finally:
        if os.path.isfile(raw_path) and raw_path != path:
            try:
                os.remove(raw_path)
            except OSError:
                pass

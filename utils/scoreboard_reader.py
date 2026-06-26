"""
scoreboard_reader.py
====================
Read scoreboard overlay from football broadcast frames.

- ``detect_scoreboard_stripe_colors`` — jersey-colour stripes beside team
  names (CV first, optional vision API fallback).
- ``detect_team_names`` — team name text via vision API (optional).
"""

from __future__ import annotations

import base64
import json
import re
import urllib.request
from typing import Any

import cv2
import numpy as np


def _encode_frame_jpeg(frame_bgr) -> str:
    """Encode a BGR numpy frame as a base64 JPEG string."""
    import cv2
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Failed to encode frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _sample_scoreboard_frame(video_frames: list, *, max_candidates: int = 5):
    """Return a few frames spread across the first ~10 % of the video."""
    n = len(video_frames)
    if n == 0:
        return []
    # Use up to max_candidates frames from the first 10 % (min 30 frames)
    window = max(30, n // 10)
    step   = max(1, window // max_candidates)
    return [video_frames[i] for i in range(0, min(window, n), step)][:max_candidates]


def _call_vision_api(
    frame_b64: str,
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
) -> dict[str, Any]:
    """Send a vision request and return the parsed JSON response body."""
    url     = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model":    model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "This is a frame from a football broadcast. "
                            "Look at the scoreboard overlay (typically at the top or bottom of the screen). "
                            "Extract the two team names shown on the scoreboard. "
                            "Return ONLY a JSON object with exactly two keys: "
                            '{"team1": "<name of left/home team>", "team2": "<name of right/away team>"}. '
                            "If you cannot find team names, return "
                            '{"team1": null, "team2": null}. '
                            "No other text."
                        ),
                    },
                ],
            }
        ],
        "max_tokens":      64,
        "temperature":     0.0,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data    = payload,
        method  = "POST",
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def detect_team_names(
    video_frames: list,
    *,
    api_key: str,
    model: str = "gpt-4o",
    base_url: str = "https://api.openai.com/v1",
    timeout: int = 30,
) -> tuple[str | None, str | None]:
    """Try to read team names from scoreboard in the first frames of the video.

    Returns
    -------
    (team1_name, team2_name) — either or both may be None if undetected.
    """
    candidates = _sample_scoreboard_frame(video_frames)
    if not candidates:
        return None, None

    for frame in candidates:
        try:
            frame_b64 = _encode_frame_jpeg(frame)
            body      = _call_vision_api(frame_b64, api_key, model, base_url, timeout)
            content   = body["choices"][0]["message"]["content"]

            # Strip markdown fences if present
            stripped = content.strip()
            fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped)
            if fence_match:
                stripped = fence_match.group(1).strip()

            data   = json.loads(stripped)
            t1_raw = data.get("team1")
            t2_raw = data.get("team2")

            t1 = str(t1_raw).strip() if t1_raw and str(t1_raw).strip() not in ("null", "None", "") else None
            t2 = str(t2_raw).strip() if t2_raw and str(t2_raw).strip() not in ("null", "None", "") else None

            if t1 or t2:
                return t1, t2

        except Exception as exc:
            print(f"[scoreboard_reader] Frame attempt failed: {exc}", flush=True)
            continue

    return None, None


# ── scoreboard jersey-stripe colours (CV + optional vision) ─────────────────

def _dominant_stripe_bgr(region: np.ndarray) -> np.ndarray | None:
    """Dominant saturated colour in a scoreboard sub-region (jersey stripe)."""
    if region.size == 0:
        return None

    flat = region.reshape(-1, 3).astype(np.float32)
    hsv  = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat  = hsv[:, :, 1].ravel()
    val  = hsv[:, :, 2].ravel()

    mask = (sat >= 45) & (val >= 35) & (val <= 250)
    pts  = flat[mask]
    if len(pts) < 25:
        return None

    lab = cv2.cvtColor(
        pts.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB,
    ).reshape(-1, 3)
    L = lab[:, 0]
    pts = pts[(L > 30) & (L < 225)]
    if len(pts) < 15:
        return None

    return np.median(pts, axis=0).astype(np.float32)


def _stripes_from_band(band: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Left / right jersey stripes inside a horizontal scoreboard band."""
    _bh, bw = band.shape[:2]
    if bw < 40:
        return None, None
    left_zone  = band[:, : max(8, int(bw * 0.38))]
    right_zone = band[:, max(0, int(bw * 0.62)) :]
    return _dominant_stripe_bgr(left_zone), _dominant_stripe_bgr(right_zone)


def _stripes_cv_single_frame(frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Sample left/right stripe colours from one broadcast frame."""
    h, w = frame.shape[:2]
    bands = [
        frame[int(h * 0.86) : h, int(w * 0.12) : int(w * 0.88)],
        frame[0 : max(12, int(h * 0.14)), int(w * 0.12) : int(w * 0.88)],
    ]

    best: tuple[np.ndarray | None, np.ndarray | None] = (None, None)
    best_sep = 0.0

    for band in bands:
        left, right = _stripes_from_band(band)
        if left is None or right is None:
            continue
        sep = float(np.linalg.norm(left - right))
        if sep > best_sep and sep >= 15.0:
            best_sep = sep
            best = (left, right)

    return best


def detect_scoreboard_stripe_colors_cv(
    video_frames: list,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read jersey-stripe colours from the scoreboard using OpenCV only."""
    for frame in _sample_scoreboard_frame(video_frames):
        left, right = _stripes_cv_single_frame(frame)
        if left is not None and right is not None:
            return left, right
    return None, None


def _call_vision_stripe_api(
    frame_b64: str,
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        "This is a football broadcast frame. On the scoreboard overlay "
                        "(top or bottom), each team name has a small coloured stripe/bar "
                        "matching that team's jersey colour.\n"
                        "Return ONLY JSON:\n"
                        '{"team1_stripe_rgb": [R,G,B], "team2_stripe_rgb": [R,G,B]}\n'
                        "team1 = left/home stripe, team2 = right/away stripe.\n"
                        "Sample the jersey stripe colour, not white text or black background.\n"
                        'If not visible: {"team1_stripe_rgb": null, "team2_stripe_rgb": null}'
                    ),
                },
            ],
        }],
        "max_tokens": 96,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_rgb_stripe(raw) -> np.ndarray | None:
    if raw is None or not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    try:
        r, g, b = (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        return None
    if not all(0 <= c <= 255 for c in (r, g, b)):
        return None
    return np.array([b, g, r], dtype=np.float32)


def detect_scoreboard_stripe_colors_vision(
    video_frames: list,
    *,
    api_key: str,
    model: str = "gpt-4o",
    base_url: str = "https://api.openai.com/v1",
    timeout: int = 30,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read jersey-stripe colours via vision LLM."""
    for frame in _sample_scoreboard_frame(video_frames):
        try:
            frame_b64 = _encode_frame_jpeg(frame)
            body    = _call_vision_stripe_api(
                frame_b64, api_key, model, base_url, timeout,
            )
            content = body["choices"][0]["message"]["content"]
            stripped = content.strip()
            fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped)
            if fence:
                stripped = fence.group(1).strip()
            data  = json.loads(stripped)
            left  = _parse_rgb_stripe(data.get("team1_stripe_rgb"))
            right = _parse_rgb_stripe(data.get("team2_stripe_rgb"))
            if left is not None and right is not None:
                return left, right
        except Exception as exc:
            print(f"[scoreboard_reader] Vision stripe attempt failed: {exc}", flush=True)
    return None, None


def detect_scoreboard_stripe_colors(
    video_frames: list,
    *,
    api_key: str | None = None,
    model: str = "gpt-4o",
    base_url: str = "https://api.openai.com/v1",
    timeout: int = 30,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect left/right jersey stripe colours on the broadcast scoreboard.

    Tries fast OpenCV sampling first, then optional vision API fallback.
    Returns (left_bgr, right_bgr) for team 1 (left) and team 2 (right).
    """
    left, right = detect_scoreboard_stripe_colors_cv(video_frames)
    if left is not None and right is not None:
        print(
            f"[scoreboard_reader] CV stripes: left={left.astype(int).tolist()} "
            f"right={right.astype(int).tolist()}",
            flush=True,
        )
        return left, right

    if api_key:
        left, right = detect_scoreboard_stripe_colors_vision(
            video_frames, api_key=api_key, model=model,
            base_url=base_url, timeout=timeout,
        )
        if left is not None and right is not None:
            print(
                f"[scoreboard_reader] Vision stripes: left={left.astype(int).tolist()} "
                f"right={right.astype(int).tolist()}",
                flush=True,
            )
            return left, right

    print("[scoreboard_reader] Could not read scoreboard stripe colours", flush=True)
    return None, None

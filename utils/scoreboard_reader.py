"""
scoreboard_reader.py
====================
Detect team names from the scoreboard overlay in a football video.

Uses the vision-capable LLM (GPT-4o or compatible) that is already
configured for TacticalNarrator.  Falls back gracefully when the API
is unavailable or the scoreboard cannot be parsed.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.request
import urllib.error
from typing import Any


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

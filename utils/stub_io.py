"""Save/load tracking stubs produced on Colab (enriched) or legacy raw stubs."""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any


def video_fingerprint(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    """MD5 of video bytes — dùng để không tái sử dụng stub của video khác."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def save_track_stub(
    path: str,
    tracks: dict,
    fps: float,
    *,
    enriched: bool = True,
    source_video: str | None = None,
) -> None:
    """Persist tracks (+ fps) after GPU tracking on Colab."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: dict[str, Any] = {
        "tracks": tracks,
        "fps": float(fps),
        "enriched": enriched,
    }
    if source_video and os.path.exists(source_video):
        payload["source_video_md5"] = video_fingerprint(source_video)
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


def stub_matches_video(stub_path: str, video_path: str) -> bool:
    """True if stub was built for this video (legacy stubs without hash → False)."""
    if not os.path.exists(stub_path) or not os.path.exists(video_path):
        return False
    with open(stub_path, "rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        return False
    expected = data.get("source_video_md5")
    if not expected:
        return False
    return expected == video_fingerprint(video_path)


def remove_track_stub(path: str) -> None:
    if path and os.path.exists(path):
        os.remove(path)


def load_track_stub(path: str) -> tuple[dict, float | None, bool]:
    """Return ``(tracks, fps, enriched)``. Supports legacy raw-tracks pickles."""
    with open(path, "rb") as fh:
        data = pickle.load(fh)

    if isinstance(data, dict) and "tracks" in data:
        return (
            data["tracks"],
            data.get("fps"),
            bool(data.get("enriched", False)),
        )

    return data, None, False

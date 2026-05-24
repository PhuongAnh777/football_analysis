"""Save/load tracking stubs produced on Colab (enriched) or legacy raw stubs."""

from __future__ import annotations

import os
import pickle
from typing import Any


def save_track_stub(path: str, tracks: dict, fps: float, *, enriched: bool = True) -> None:
    """Persist tracks (+ fps) after GPU tracking on Colab."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: dict[str, Any] = {
        "tracks": tracks,
        "fps": float(fps),
        "enriched": enriched,
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


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

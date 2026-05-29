"""Lưu metadata + kết quả job ra disk (sống sót sau khi uvicorn reload)."""

from __future__ import annotations

import json
import os
from typing import Any

_META_FILE = "job_meta.json"
_RESULT_FILE = "job_result.json"


def _meta_path(output_dir: str) -> str:
    return os.path.join(output_dir, _META_FILE)


def _result_path(output_dir: str) -> str:
    return os.path.join(output_dir, _RESULT_FILE)


def save_job_meta(output_dir: str, payload: dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(_meta_path(output_dir), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_job_meta(job_id: str, output_root: str) -> dict[str, Any] | None:
    job_dir = os.path.join(output_root, job_id)
    path = _meta_path(job_dir)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    # Job cũ (trước khi có persistence): suy ra từ file output còn trên disk
    for name in ("output_video_h264.mp4", "output_video.mp4", "output_video.avi"):
        video = os.path.join(job_dir, name)
        if os.path.isfile(video):
            return {
                "job_id": job_id,
                "status": "done",
                "video_path": os.path.abspath(video),
            }
    return None


def save_job_result(output_dir: str, result: dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(_result_path(output_dir), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)


def load_job_result(job_id: str, output_root: str) -> dict[str, Any] | None:
    path = _result_path(os.path.join(output_root, job_id))
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

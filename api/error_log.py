"""Ghi log lỗi pipeline ra file — dễ xem/copy sau khi job fail."""

from __future__ import annotations

import os
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")
_GLOBAL_LOG = os.path.join(_LOGS_DIR, "pipeline_errors.log")
_ERROR_FILENAME = "error.log"


def job_error_log_path(output_dir: str) -> str:
    return os.path.join(output_dir, _ERROR_FILENAME)


def save_job_error(
    job_id: str,
    error_detail: str,
    *,
    output_dir: str,
    step_key: str = "",
    source: str = "pipeline",
) -> str:
    """
    Ghi full traceback vào output_videos/{job_id}/error.log
    và append một dòng tóm tắt vào logs/pipeline_errors.log.
    Returns absolute path to error.log.
    """
    os.makedirs(output_dir, exist_ok=True)
    per_job_path = job_error_log_path(output_dir)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    header = (
        f"job_id: {job_id}\n"
        f"time: {ts}\n"
        f"source: {source}\n"
        f"step_key: {step_key or 'n/a'}\n"
        f"{'=' * 72}\n"
    )
    with open(per_job_path, "w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(error_detail)
        if not error_detail.endswith("\n"):
            fh.write("\n")

    os.makedirs(_LOGS_DIR, exist_ok=True)
    first_line = error_detail.strip().splitlines()[0] if error_detail.strip() else "unknown"
    with open(_GLOBAL_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] job={job_id} source={source} | {first_line}\n")
        fh.write(f"  → {os.path.abspath(per_job_path)}\n")

    abs_path = os.path.abspath(per_job_path)
    print(f"[error] job={job_id} log saved → {abs_path}", flush=True)
    return abs_path

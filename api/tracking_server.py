"""
tracking_server.py
==================
Lightweight FastAPI — chỉ chạy YOLO tracking trên GPU (Colab).
Local backend gọi qua ngrok; pipeline còn lại chạy trên máy local.

Run on Colab:
    uvicorn api.tracking_server:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback as tb
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_INPUT_VIDEO = os.path.join(_PROJECT_ROOT, "input_videos", "input_video.mp4")
_STUB_PATH = os.path.join(_PROJECT_ROOT, "stubs", "track_stubs.pkl")
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "best.pt")
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024
_PORT = int(os.getenv("TRACKING_SERVER_PORT", "8001"))

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="colab-track")


@dataclass
class TrackJob:
    job_id: str
    status: str = "processing"  # processing | done | error
    current_step: str = "Khởi tạo..."
    error: Optional[str] = None
    stub_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


_jobs: Dict[str, TrackJob] = {}


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Football Analysis — Colab Tracking",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_job(job_id: str) -> TrackJob:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


def _run_tracking_job(job_id: str, video_path: str) -> None:
    job = _jobs[job_id]
    try:
        from scripts.run_tracking import run_tracking

        job.current_step = "Đang đọc video & tracking (GPU)..."
        os.makedirs(os.path.dirname(_STUB_PATH), exist_ok=True)
        run_tracking(
            video_path,
            model_path=_MODEL_PATH,
            stub_path=_STUB_PATH,
        )
        if not os.path.exists(_STUB_PATH):
            raise FileNotFoundError(f"Stub not created: {_STUB_PATH}")

        job.stub_path = _STUB_PATH
        job.status = "done"
        job.current_step = "Tracking hoàn thành"
    except Exception as exc:
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}\n{tb.format_exc()}"
        job.current_step = "Lỗi"


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "colab-tracking",
        "model_loaded": os.path.exists(_MODEL_PATH),
    }


@app.post("/api/track", status_code=202)
async def track(video: UploadFile = File(...)):
    """Nhận video từ local backend → chạy tracking GPU → lưu stub."""
    if video.content_type and video.content_type not in (
        "video/mp4",
        "video/avi",
        "video/x-msvideo",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=415,
            detail="Unsupported media type. Upload MP4 or AVI.",
        )

    contents = await video.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {_MAX_UPLOAD_BYTES // (1024**2)} MB).",
        )

    os.makedirs(os.path.dirname(_INPUT_VIDEO), exist_ok=True)
    async with aiofiles.open(_INPUT_VIDEO, "wb") as fh:
        await fh.write(contents)

    import uuid

    job_id = str(uuid.uuid4())
    _jobs[job_id] = TrackJob(job_id=job_id, current_step="Đã nhận video, chờ GPU...")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_tracking_job, job_id, _INPUT_VIDEO)

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/track/status/{job_id}")
async def track_status(job_id: str):
    job = _require_job(job_id)
    return {
        "job_id": job_id,
        "status": job.status,
        "current_step": job.current_step,
        "error": job.error,
    }


@app.get("/api/track/stub/{job_id}")
async def track_stub(job_id: str):
    """Tải file stub (.pkl) khi tracking xong."""
    job = _require_job(job_id)

    if job.status == "processing":
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "processing",
                "current_step": job.current_step,
            },
        )

    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.error or "Tracking failed.")

    stub_path = job.stub_path or _STUB_PATH
    if not os.path.exists(stub_path):
        raise HTTPException(status_code=404, detail="Stub file not found.")

    return FileResponse(
        stub_path,
        media_type="application/octet-stream",
        filename="track_stubs.pkl",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.tracking_server:app",
        host="0.0.0.0",
        port=_PORT,
        log_level="info",
    )

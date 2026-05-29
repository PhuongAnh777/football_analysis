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

_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "best.pt")
_INPUT_DIR = os.path.join(_PROJECT_ROOT, "input_videos")
_STUB_DIR = os.path.join(_PROJECT_ROOT, "stubs")
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
    video_path: Optional[str] = None
    video_bytes: int = 0
    video_md5: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None


_jobs: Dict[str, TrackJob] = {}


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def _job_video_path(job_id: str) -> str:
    return os.path.join(_INPUT_DIR, f"{job_id}.mp4")


def _job_stub_path(job_id: str) -> str:
    return os.path.join(_STUB_DIR, f"{job_id}_track_stubs.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Football Analysis — Colab Tracking",
    version="1.1.0",
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


def _run_tracking_job(job_id: str, video_path: str, stub_path: str) -> None:
    job = _jobs[job_id]
    try:
        import torch
        from scripts.run_tracking import run_tracking
        from utils.stub_io import video_fingerprint

        gpu = torch.cuda.is_available()
        print(
            f"[colab-track] job={job_id} GPU={gpu} "
            f"video={video_path} ({job.video_bytes:,} bytes md5={job.video_md5})",
            flush=True,
        )
        if not gpu:
            raise RuntimeError("Colab chưa bật GPU — Runtime → Change runtime type → T4 GPU")

        job.current_step = "Đang tracking YOLO trên GPU..."
        os.makedirs(_STUB_DIR, exist_ok=True)
        if os.path.exists(stub_path):
            os.remove(stub_path)

        run_tracking(
            video_path,
            model_path=_MODEL_PATH,
            stub_path=stub_path,
        )
        if not os.path.exists(stub_path):
            raise FileNotFoundError(f"Stub not created: {stub_path}")

        print(
            f"[colab-track] job={job_id} DONE stub={stub_path} "
            f"({os.path.getsize(stub_path):,} bytes)",
            flush=True,
        )

        job.stub_path = stub_path
        job.status = "done"
        job.current_step = "Tracking hoàn thành"
        job.finished_at = datetime.utcnow()
    except Exception as exc:
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}\n{tb.format_exc()}"
        job.current_step = "Lỗi"
        job.finished_at = datetime.utcnow()
        print(f"[colab-track] job={job_id} ERROR: {exc}", flush=True)


@app.get("/api/health")
async def health():
    try:
        import torch
        gpu = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu else None
    except ImportError:
        gpu, gpu_name = False, None

    return {
        "status": "ok",
        "service": "colab-tracking",
        "model_loaded": os.path.exists(_MODEL_PATH),
        "gpu_available": gpu,
        "gpu_name": gpu_name,
    }


@app.post("/api/track", status_code=202)
async def track(video: UploadFile = File(...)):
    """Nhận video từ local backend → chạy tracking GPU → lưu stub riêng theo job."""
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

    import uuid
    from utils.stub_io import video_fingerprint

    job_id = str(uuid.uuid4())
    video_path = _job_video_path(job_id)
    stub_path = _job_stub_path(job_id)

    os.makedirs(_INPUT_DIR, exist_ok=True)
    async with aiofiles.open(video_path, "wb") as fh:
        await fh.write(contents)

    video_md5 = video_fingerprint(video_path)
    print(
        f"[colab-track] POST job={job_id} received {len(contents):,} bytes md5={video_md5}",
        flush=True,
    )

    _jobs[job_id] = TrackJob(
        job_id=job_id,
        current_step="Đã nhận video, chờ GPU...",
        video_path=video_path,
        video_bytes=len(contents),
        video_md5=video_md5,
    )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_tracking_job, job_id, video_path, stub_path)

    return {
        "job_id": job_id,
        "status": "processing",
        "video_bytes": len(contents),
        "video_md5": video_md5,
    }


@app.get("/api/track/status/{job_id}")
async def track_status(job_id: str):
    job = _require_job(job_id)
    elapsed = (datetime.utcnow() - job.started_at).total_seconds()
    return {
        "job_id": job_id,
        "status": job.status,
        "current_step": job.current_step,
        "error": job.error,
        "video_bytes": job.video_bytes,
        "video_md5": job.video_md5,
        "elapsed_sec": round(elapsed, 1),
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

    stub_path = job.stub_path or _job_stub_path(job_id)
    if not os.path.exists(stub_path):
        raise HTTPException(status_code=404, detail="Stub file not found.")

    return FileResponse(
        stub_path,
        media_type="application/octet-stream",
        filename=f"{job_id}_track_stubs.pkl",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.tracking_server:app",
        host="0.0.0.0",
        port=_PORT,
        log_level="info",
    )

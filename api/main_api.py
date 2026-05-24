"""
main_api.py
===========
FastAPI application that exposes the Football Analysis pipeline as a
REST API.

Endpoints
---------
POST   /api/analyze           Upload video → start background pipeline
GET    /api/status/{job_id}   Poll progress
GET    /api/results/{job_id}  Retrieve full results (charts + reports)
GET    /api/video/{job_id}    Stream output video (MP4 / AVI)
GET    /api/health            Service health check

Run:
    uvicorn api.main_api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api.job_store import cleanup_old_jobs, create_job, get_job, jobs
from api.pipeline_runner import run_pipeline

# ── Config ────────────────────────────────────────────────────────────────────
_INPUT_VIDEO_PATH = os.path.join(_PROJECT_ROOT, "input_videos", "input_video.mp4")
_MAX_UPLOAD_BYTES  = 500 * 1024 * 1024   # 500 MB
_PIPELINE_TIMEOUT  = 30 * 60             # 30 minutes
_CLEANUP_INTERVAL  = 30 * 60             # 30 minutes


def _parse_cors_origins() -> list[str]:
    """Comma-separated origins, e.g. CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000"""
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["http://localhost:3000"]


_ALLOWED_ORIGINS = _parse_cors_origins()

# Single-thread executor so the CPU-heavy pipeline doesn't block the event loop
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background cleanup task on startup."""
    cleanup_task = asyncio.create_task(_cleanup_loop())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


async def _cleanup_loop() -> None:
    """Periodically remove jobs older than 2 hours."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL)
        removed = cleanup_old_jobs(max_age_seconds=7200)
        if removed:
            print(f"[cleanup] Removed {removed} expired job(s).")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title        = "Football Analysis API",
    version      = "1.0.0",
    description  = "REST API wrapper for the Football Analysis pipeline.",
    lifespan     = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _require_job(job_id: str):
    """Return the job or raise 404."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


def _model_loaded() -> bool:
    """Check whether the YOLO model file exists on disk."""
    return os.path.exists(os.path.join(_PROJECT_ROOT, "models", "best.pt"))


def _track_stub_loaded() -> bool:
    stub = os.getenv("TRACK_STUB_PATH", os.path.join(_PROJECT_ROOT, "stubs", "track_stubs.pkl"))
    return os.path.exists(stub)


# ── Endpoints ─────────────────────────────────────────────────────────────────

# ── POST /api/analyze ─────────────────────────────────────────────────────────

@app.post("/api/analyze", status_code=202)
async def analyze(video: UploadFile = File(...)):
    """Accept a video file and start the analysis pipeline.

    Returns a ``job_id`` that can be used to poll status and retrieve
    results once the pipeline completes.
    """
    if video.content_type not in ("video/mp4", "video/avi", "video/x-msvideo"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported media type. Upload an MP4 or AVI file.",
        )

    contents = await video.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {_MAX_UPLOAD_BYTES // (1024**2)} MB.",
        )

    os.makedirs(os.path.dirname(_INPUT_VIDEO_PATH), exist_ok=True)
    async with aiofiles.open(_INPUT_VIDEO_PATH, "wb") as fh:
        await fh.write(contents)

    job_id = create_job()

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        run_pipeline,
        _INPUT_VIDEO_PATH,
        job_id,
        jobs,
    )

    return {"job_id": job_id, "status": "processing"}


# ── GET /api/status/{job_id} ──────────────────────────────────────────────────

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    """Return the current progress of a pipeline job."""
    job = _require_job(job_id)
    return {
        "job_id":       job.job_id,
        "status":       job.status,
        "progress":     round(job.progress, 4),
        "current_step": job.current_step,
        "step_key":     job.step_key,
        "error":        job.error,
    }


# ── GET /api/results/{job_id} ─────────────────────────────────────────────────

@app.get("/api/results/{job_id}")
async def results(job_id: str):
    """Return the full analysis results once the pipeline is done.

    The ``charts`` field contains base64-encoded PNG strings.
    """
    job = _require_job(job_id)

    if job.status == "processing":
        return JSONResponse(
            status_code=202,
            content={
                "job_id":       job.job_id,
                "status":       "processing",
                "progress":     round(job.progress, 4),
                "current_step": job.current_step,
            },
        )

    if job.status == "error":
        raise HTTPException(
            status_code=500,
            detail={"job_id": job.job_id, "error": job.error},
        )

    result = job.result or {}
    return {
        "job_id":           job.job_id,
        "evaluation":       result.get("evaluation", {}),
        "match_report":     result.get("match_report", {}),
        "charts":           result.get("charts", {}),
        "teams":            result.get("teams", []),
        "players":          result.get("players", []),
        "timeline":         result.get("timeline", []),
        "notable_players":  result.get("notable_players", {}),
        "fps":              result.get("fps", 24),
        "video_url":        f"/api/video/{job_id}",
    }


# ── GET /api/video/{job_id} ───────────────────────────────────────────────────

@app.get("/api/video/{job_id}")
async def video(job_id: str, request: Request):
    """Stream the output video with HTTP range-request support.

    The server attempts to serve an MP4 file (converted via ffmpeg when
    available); if only the AVI exists it falls back to ``video/x-msvideo``.
    """
    job = _require_job(job_id)

    if job.status != "done":
        raise HTTPException(
            status_code=425,
            detail="Video is not ready yet. Check /api/status/{job_id}.",
        )

    video_path = job.video_path
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found on server.")

    media_type   = "video/mp4" if video_path.endswith(".mp4") else "video/x-msvideo"
    file_size    = os.path.getsize(video_path)
    range_header = request.headers.get("range")

    # ── Range request (partial content) ──────────────────────────────────────
    if range_header:
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header.strip())
        if match:
            start = int(match.group(1))
            end   = int(match.group(2)) if match.group(2) else file_size - 1
            end   = min(end, file_size - 1)

            if start > end or start >= file_size:
                raise HTTPException(
                    status_code=416,
                    detail="Requested range not satisfiable.",
                )

            chunk_size = end - start + 1

            async def _stream_range():
                async with aiofiles.open(video_path, "rb") as fh:
                    await fh.seek(start)
                    remaining = chunk_size
                    while remaining > 0:
                        read_size = min(65536, remaining)
                        data = await fh.read(read_size)
                        if not data:
                            break
                        remaining -= len(data)
                        yield data

            return StreamingResponse(
                _stream_range(),
                status_code   = 206,
                media_type    = media_type,
                headers       = {
                    "Content-Range":  f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges":  "bytes",
                    "Content-Length": str(chunk_size),
                    "Cache-Control":  "no-cache",
                },
            )

    # ── Full file ─────────────────────────────────────────────────────────────
    async def _stream_full():
        async with aiofiles.open(video_path, "rb") as fh:
            while True:
                chunk = await fh.read(65536)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        _stream_full(),
        media_type = media_type,
        headers    = {
            "Accept-Ranges":  "bytes",
            "Content-Length": str(file_size),
            "Cache-Control":  "no-cache",
        },
    )


# ── GET /api/health ───────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Return a simple liveness / readiness response."""
    return {"status": "ok", "model_loaded": _model_loaded(), "track_stub_loaded": _track_stub_loaded()}


# ── Dev entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main_api:app",
        host               = "0.0.0.0",
        port               = 8000,
        reload             = True,
        timeout_keep_alive = _PIPELINE_TIMEOUT,
    )

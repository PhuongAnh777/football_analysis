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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_dotenv() -> None:
    """Load repo-root `.env` into os.environ (does not override existing vars)."""
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value


_load_dotenv()

from api.job_store import cleanup_old_jobs, create_job, get_job, jobs
from api.job_persistence import load_job_meta, load_job_result
from api.pipeline_runner import _convert_to_mp4, _job_output_dir, _job_stub_path, run_pipeline
from api.error_log import job_error_log_path, save_job_error
from utils.stub_io import remove_track_stub, video_fingerprint
from utils.video_utils import ensure_browser_playable

_OUTPUT_VIDEOS_ROOT = os.path.join(_PROJECT_ROOT, "output_videos")

_DEFAULT_TRACK_STUB = os.path.join(_PROJECT_ROOT, "stubs", "track_stubs.pkl")
_INPUT_VIDEO_PATH = os.path.join(_PROJECT_ROOT, "input_videos", "input_video.mp4")

# ── Config ────────────────────────────────────────────────────────────────────
_MAX_UPLOAD_BYTES  = 500 * 1024 * 1024   # 500 MB
_PIPELINE_TIMEOUT  = 30 * 60             # 30 minutes
_CLEANUP_INTERVAL  = 30 * 60             # 30 minutes


_JOB_INPUT_MP4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.mp4$",
    re.IGNORECASE,
)


def _cleanup_per_job_input_copies() -> int:
    """Xóa input_videos/{job_id}.mp4 do bản code cũ tạo (không đụng file test khác)."""
    input_dir = os.path.dirname(_INPUT_VIDEO_PATH)
    if not os.path.isdir(input_dir):
        return 0
    removed = 0
    for name in os.listdir(input_dir):
        if not _JOB_INPUT_MP4.match(name):
            continue
        try:
            os.remove(os.path.join(input_dir, name))
            removed += 1
        except OSError:
            pass
    return removed


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


def _get_job_or_disk(job_id: str):
    """Job trong RAM, hoặc metadata đã lưu disk sau khi server reload."""
    job = get_job(job_id)
    if job is not None:
        return job, False
    meta = load_job_meta(job_id, _OUTPUT_VIDEOS_ROOT)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return meta, True


def _model_loaded() -> bool:
    """Check whether the YOLO model file exists on disk."""
    return os.path.exists(os.path.join(_PROJECT_ROOT, "models", "best.pt"))


def _track_stub_loaded() -> bool:
    stub = os.getenv("TRACK_STUB_PATH", _DEFAULT_TRACK_STUB)
    return os.path.exists(stub)


def _resolve_job_video_path(job_id: str, stored_path: str | None) -> str | None:
    """Ưu tiên MP4 trong thư mục job (trình duyệt không phát AVI XVID ổn định)."""
    job_dir = os.path.abspath(_job_output_dir(job_id))
    if not os.path.isdir(job_dir):
        return None

    def _in_job_dir(path: str) -> bool:
        abs_path = os.path.abspath(path)
        prefix = os.path.normcase(job_dir + os.sep)
        return os.path.normcase(abs_path).startswith(prefix)

    for name in ("output_video_h264.mp4", "output_video.mp4", "output_video.avi"):
        path = os.path.join(job_dir, name)
        if os.path.isfile(path):
            return path

    if stored_path and os.path.isfile(stored_path) and _in_job_dir(stored_path):
        return os.path.abspath(stored_path)
    return None


def _ensure_browser_video(path: str) -> str:
    """Ensure MP4 is H.264 so Chrome/Edge can play it in <video>."""
    return ensure_browser_playable(path)


def _colab_tracking_url() -> str | None:
    _load_dotenv()  # đọc lại .env (uvicorn --reload không theo dõi file .env)
    url = os.getenv("COLAB_TRACKING_URL", "").strip()
    return url or None


def _run_job_with_optional_colab(video_path: str, job_id: str, team_names: dict | None = None) -> None:
    """Colab GPU tracking (optional) → local pipeline."""
    colab_url = _colab_tracking_url()
    stub_path = _job_stub_path(job_id)
    use_stub = False

    remove_track_stub(stub_path)
    remove_track_stub(os.getenv("TRACK_STUB_PATH", _DEFAULT_TRACK_STUB))

    if colab_url:
        print(f"[analyze] Colab GPU tracking → {colab_url}", flush=True)
        from api.colab_tracking_client import fetch_colab_tracking_stub

        job = jobs[job_id]

        def _on_colab_progress(msg: str) -> None:
            job.current_step = msg
            job.step_key = "tracking_remote"
            job.progress = 0.05

        try:
            fetch_colab_tracking_stub(
                video_path,
                colab_url,
                stub_path,
                on_progress=_on_colab_progress,
            )
            use_stub = True
        except Exception as exc:
            import traceback as _tb

            job = jobs[job_id]
            error_detail = f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}"
            job.status = "error"
            job.step_key = "error"
            job.current_step = "Lỗi tracking Colab"
            job.error = error_detail
            job.error_log_path = save_job_error(
                job_id,
                error_detail,
                output_dir=_job_output_dir(job_id),
                step_key="tracking_remote",
                source="colab",
            )
            from api.job_persistence import save_job_meta

            save_job_meta(
                _job_output_dir(job_id),
                {
                    "job_id": job_id,
                    "status": "error",
                    "error": error_detail,
                    "error_log_path": job.error_log_path,
                    "input_path": job.input_path,
                    "input_md5": job.input_md5,
                    "input_filename": job.input_filename,
                    "input_size_bytes": job.input_size_bytes,
                },
            )
            return
    else:
        print(
            "[analyze] COLAB_TRACKING_URL trống — tracking chạy trên máy local (GPU Colab không dùng). "
            "Điền .env rồi upload lại.",
            flush=True,
        )

    run_pipeline(
        video_path,
        job_id,
        jobs,
        track_stub_path=stub_path,
        use_track_stub=use_stub,
        team_names=team_names,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

# ── POST /api/analyze ─────────────────────────────────────────────────────────

@app.post("/api/analyze", status_code=202)
async def analyze(
    video: UploadFile = File(...),
    team1_name: Optional[str] = Form(default=None),
    team2_name: Optional[str] = Form(default=None),
):
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

    job_id = create_job()
    n_removed = _cleanup_per_job_input_copies()
    if n_removed:
        print(f"[analyze] Đã xóa {n_removed} file input video cũ (theo job_id).", flush=True)

    os.makedirs(os.path.dirname(_INPUT_VIDEO_PATH), exist_ok=True)
    async with aiofiles.open(_INPUT_VIDEO_PATH, "wb") as fh:
        await fh.write(contents)

    input_md5 = video_fingerprint(_INPUT_VIDEO_PATH)
    job = jobs[job_id]
    job.input_path = _INPUT_VIDEO_PATH
    job.input_md5 = input_md5
    job.input_size_bytes = len(contents)
    job.input_filename = video.filename or os.path.basename(_INPUT_VIDEO_PATH)

    # Build team name dict from required form fields
    _team_names: dict[int, str] = {}
    if not team1_name or not team1_name.strip():
        raise HTTPException(
            status_code=400,
            detail="Vui lòng nhập tên đội 1.",
        )
    if not team2_name or not team2_name.strip():
        raise HTTPException(
            status_code=400,
            detail="Vui lòng nhập tên đội 2.",
        )
    _team_names[1] = team1_name.strip()
    _team_names[2] = team2_name.strip()

    print(
        f"[analyze] job={job_id} file={job.input_filename!r} "
        f"({len(contents):,} bytes, md5={input_md5}) team_names={_team_names} → {_INPUT_VIDEO_PATH}",
        flush=True,
    )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_job_with_optional_colab,
        _INPUT_VIDEO_PATH,
        job_id,
        _team_names,
    )

    return {"job_id": job_id, "status": "processing"}


# ── GET /api/status/{job_id} ──────────────────────────────────────────────────

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    """Return the current progress of a pipeline job."""
    job = _require_job(job_id)
    error_log = getattr(job, "error_log_path", None)
    if not error_log and job.status == "error":
        candidate = job_error_log_path(_job_output_dir(job_id))
        if os.path.isfile(candidate):
            error_log = os.path.abspath(candidate)
    return {
        "job_id":       job.job_id,
        "status":       job.status,
        "progress":     round(job.progress, 4),
        "current_step": job.current_step,
        "step_key":     job.step_key,
        "error":        job.error,
        "error_log_path": error_log,
        "input_filename": job.input_filename,
        "input_md5":    job.input_md5,
        "input_size_bytes": job.input_size_bytes,
    }


# ── GET /api/results/{job_id} ─────────────────────────────────────────────────

@app.get("/api/results/{job_id}")
async def results(job_id: str):
    """Return the full analysis results once the pipeline is done.

    The ``charts`` field contains base64-encoded PNG strings.
    """
    job_or_meta, from_disk = _get_job_or_disk(job_id)

    if from_disk:
        if job_or_meta.get("status") != "done":
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not ready.")
        result = load_job_result(job_id, _OUTPUT_VIDEOS_ROOT) or {}
        return {
            "job_id": job_id,
            "input_filename": job_or_meta.get("input_filename", ""),
            "input_md5": job_or_meta.get("input_md5"),
            "input_size_bytes": job_or_meta.get("input_size_bytes", 0),
            "evaluation": result.get("evaluation", {}),
            "match_report": result.get("match_report", {}),
            "charts": result.get("charts", {}),
            "teams": result.get("teams", []),
            "players": result.get("players", []),
            "timeline": result.get("timeline", []),
            "notable_players": result.get("notable_players", {}),
            "fps": result.get("fps", 24),
            "video_url": f"/api/video/{job_id}",
        }

    job = job_or_meta

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
        "input_filename":   job.input_filename,
        "input_md5":        job.input_md5,
        "input_size_bytes": job.input_size_bytes,
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
    job_or_meta, from_disk = _get_job_or_disk(job_id)

    status = job_or_meta.get("status") if from_disk else job_or_meta.status
    if status != "done":
        raise HTTPException(
            status_code=425,
            detail="Video is not ready yet. Check /api/status/{job_id}.",
        )

    stored = job_or_meta.get("video_path") if from_disk else job_or_meta.video_path
    video_path = _resolve_job_video_path(job_id, stored)
    if not video_path:
        raise HTTPException(
            status_code=404,
            detail=f"Video not found for job '{job_id}'. Run analysis again.",
        )

    video_path = _ensure_browser_video(video_path)
    media_type = "video/mp4" if video_path.lower().endswith(".mp4") else "video/x-msvideo"
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
            "Cache-Control":  "no-store, no-cache, must-revalidate",
        },
    )


# ── GET /api/health ───────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Return a simple liveness / readiness response."""
    return {
        "status": "ok",
        "model_loaded": _model_loaded(),
        "track_stub_loaded": _track_stub_loaded(),
        "colab_tracking_url": _colab_tracking_url(),
    }


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

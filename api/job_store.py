"""
In-memory job store for the Football Analysis API.

Stores JobState objects keyed by job_id (UUID string).
Thread-safe via a module-level threading.Lock for mutations.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class JobState:
    job_id: str
    status: str              # "processing" | "done" | "error"
    progress: float = 0.0    # 0.0 – 1.0
    current_step: str = ""
    step_key: str = ""
    error: Optional[str] = None
    result: Optional[dict] = None       # populated when status == "done"
    video_path: Optional[str] = None    # path to output video file
    created_at: datetime = field(default_factory=datetime.utcnow)


# ── Global store ─────────────────────────────────────────────────────────────

_store_lock = threading.Lock()
jobs: Dict[str, JobState] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_job() -> str:
    """Create a new job entry and return its job_id."""
    job_id = str(uuid.uuid4())
    with _store_lock:
        jobs[job_id] = JobState(
            job_id=job_id,
            status="processing",
            current_step="Khởi tạo...",
        )
    return job_id


def get_job(job_id: str) -> Optional[JobState]:
    """Return the JobState for *job_id*, or None if not found."""
    return jobs.get(job_id)


def cleanup_old_jobs(max_age_seconds: int = 7200) -> int:
    """Remove jobs older than *max_age_seconds* (default 2 hours).

    Returns the number of jobs removed.
    """
    now = datetime.utcnow()
    to_delete = [
        jid
        for jid, job in jobs.items()
        if (now - job.created_at).total_seconds() > max_age_seconds
    ]
    with _store_lock:
        for jid in to_delete:
            jobs.pop(jid, None)
    return len(to_delete)

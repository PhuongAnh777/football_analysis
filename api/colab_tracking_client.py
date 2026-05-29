"""
Gọi Colab tracking server (ngrok) từ local backend sau khi user upload video trên web.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import requests

_DEFAULT_POLL_SEC = 3.0
_DEFAULT_TIMEOUT_SEC = 60 * 60  # 1 hour


def _headers() -> dict[str, str]:
    return {"ngrok-skip-browser-warning": "true"}


def _normalize_base(url: str) -> str:
    return url.rstrip("/")


def fetch_colab_tracking_stub(
    video_path: str,
    colab_base_url: str,
    dest_stub_path: str,
    *,
    on_progress: Optional[Callable[[str], None]] = None,
    poll_interval: float = _DEFAULT_POLL_SEC,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
) -> str:
    """
    Upload *video_path* lên Colab, chờ tracking xong, tải stub về *dest_stub_path*.
    Returns path to saved stub.
    """
    base = _normalize_base(colab_base_url)
    os.makedirs(os.path.dirname(os.path.abspath(dest_stub_path)), exist_ok=True)

    def _notify(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    _notify("Đang gửi video lên Colab GPU...")
    print(f"[colab] POST {base}/api/track ← {video_path}", flush=True)
    with open(video_path, "rb") as fh:
        resp = requests.post(
            f"{base}/api/track",
            files={"video": (os.path.basename(video_path), fh, "video/mp4")},
            headers=_headers(),
            timeout=600,
        )
    resp.raise_for_status()
    payload = resp.json()
    job_id = payload["job_id"]

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        status_resp = requests.get(
            f"{base}/api/track/status/{job_id}",
            headers=_headers(),
            timeout=60,
        )
        status_resp.raise_for_status()
        data = status_resp.json()

        if data.get("status") == "error":
            raise RuntimeError(data.get("error") or "Colab tracking failed.")

        if data.get("status") == "done":
            break

        step = data.get("current_step") or "Đang tracking trên Colab..."
        _notify(step)
        time.sleep(poll_interval)
    else:
        raise TimeoutError(
            f"Colab tracking timed out after {int(timeout_sec)}s. "
            "Giữ tab Colab mở và kiểm tra GPU runtime."
        )

    _notify("Đang tải stub từ Colab...")
    stub_resp = requests.get(
        f"{base}/api/track/stub/{job_id}",
        headers=_headers(),
        timeout=600,
        stream=True,
    )

    if stub_resp.status_code == 202:
        raise RuntimeError("Colab stub not ready yet — retry polling.")

    stub_resp.raise_for_status()

    tmp_path = dest_stub_path + ".part"
    with open(tmp_path, "wb") as out:
        for chunk in stub_resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                out.write(chunk)

    size = os.path.getsize(tmp_path)
    if size < 500_000:
        os.remove(tmp_path)
        raise RuntimeError(
            f"Stub từ Colab quá nhỏ ({size:,} bytes) — tracking có thể lỗi."
        )

    os.replace(tmp_path, dest_stub_path)
    _notify("Đã nhận stub từ Colab.")
    return dest_stub_path

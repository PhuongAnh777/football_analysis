"""
Gọi Colab tracking server (ngrok) từ local backend sau khi user upload video trên web.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Callable, Optional

import requests

from utils.stub_io import stub_matches_video, video_fingerprint

_DEFAULT_POLL_SEC = 3.0
_DEFAULT_TIMEOUT_SEC = 60 * 60  # 1 hour
_MIN_TRACKING_SEC = 15.0  # video dài — tracking GPU không thể xong trong vài giây


def _headers() -> dict[str, str]:
    return {"ngrok-skip-browser-warning": "true"}


def _normalize_base(url: str) -> str:
    return url.rstrip("/")


def _response_detail(resp: requests.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict):
            detail = data.get("detail", data)
            return str(detail)[:4000]
    except Exception:
        pass
    return (resp.text or "")[:4000]


def _ngrok_offline_hint(resp: requests.Response) -> str | None:
    """Detect ngrok 'endpoint is offline' HTML (404) when Colab tab/runtime đã tắt."""
    text = resp.text or ""
    m = re.search(r'data-payload="([^"]+)"', text)
    if not m:
        return None
    try:
        payload = json.loads(base64.b64decode(m.group(1)).decode())
    except Exception:
        return None
    msg = str(payload.get("message", "")).strip()
    if "offline" in msg.lower() or payload.get("code") == "3200":
        return msg or "Ngrok tunnel offline."
    return None


def _raise_colab_http_error(resp: requests.Response, label: str) -> None:
    ngrok_offline = _ngrok_offline_hint(resp)
    if ngrok_offline:
        raise RuntimeError(
            f"Colab/ngrok offline ({ngrok_offline}).\n"
            "→ Mở lại tab Colab, chạy lại cell server (Cell 5).\n"
            "→ Copy URL ngrok mới vào .env (COLAB_TRACKING_URL=...).\n"
            "→ Restart backend: uvicorn api.main_api:app --reload --port 8000\n"
            "→ Giữ tab Colab mở trong suốt quá trình upload."
        )

    detail = _response_detail(resp).strip()
    if not detail or detail == "Internal Server Error":
        detail = (
            "Colab/ngrok không trả lỗi chi tiết. Trên Colab: (1) git pull origin Collab "
            "(2) chạy lại cell server (3) kiểm tra models/best.pt (4) xem log đỏ trong cell "
            "(5) thử video ngắn hơn ~20MB nếu file ~50MB."
        )
    raise RuntimeError(f"Colab {label} lỗi {resp.status_code}: {detail}")


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
    local_md5 = video_fingerprint(video_path)
    local_size = os.path.getsize(video_path)

    def _notify(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    health = requests.get(f"{base}/api/health", headers=_headers(), timeout=30)
    if not health.ok:
        _raise_colab_http_error(health, "/api/health")
    try:
        health_data = health.json()
    except Exception:
        raise RuntimeError(
            f"Colab health endpoint trả về response không phải JSON (body rỗng hoặc HTML).\n"
            f"Nguyên nhân có thể: ngrok tunnel hết hạn, Colab session đã đóng, hoặc URL sai.\n"
            f"→ Kiểm tra lại COLAB_TRACKING_URL trong .env hoặc bỏ trống để dùng CPU local.\n"
            f"URL hiện tại: {base}\n"
            f"Response: {health.text[:500]!r}"
        )
    if health_data.get("service") != "colab-tracking":
        raise RuntimeError(
            f"URL không phải Colab tracking server: {health_data!r}"
        )

    gpu_flag = health_data.get("gpu_available")
    if gpu_flag is False:
        raise RuntimeError(
            "Colab báo GPU không khả dụng. Runtime → Change runtime type → T4 GPU "
            "→ chạy lại cell server (và git pull code mới)."
        )
    if gpu_flag is None:
        print(
            "[colab] Cảnh báo: Colab server chưa có trường gpu_available — "
            "chạy `git pull` trên Colab rồi restart cell server. Tiếp tục thử tracking...",
            flush=True,
        )

    print(
        f"[colab] health OK gpu={health_data.get('gpu_name', 'n/a')} "
        f"local_video md5={local_md5} size={local_size:,}",
        flush=True,
    )

    if local_size > 40 * 1024 * 1024:
        print(
            "[colab] Cảnh báo: video > 40MB — upload qua ngrok có thể lỗi. "
            "Nên dùng clip ngắn hơn hoặc git pull Colab mới nhất.",
            flush=True,
        )

    _notify("Đang gửi video lên Colab GPU...")
    print(f"[colab] POST {base}/api/track ← {video_path}", flush=True)
    with open(video_path, "rb") as fh:
        resp = requests.post(
            f"{base}/api/track",
            files={"video": (os.path.basename(video_path), fh, "video/mp4")},
            headers=_headers(),
            timeout=600,
        )
    if not resp.ok:
        _raise_colab_http_error(resp, "/api/track")
    try:
        payload = resp.json()
    except ValueError:
        _raise_colab_http_error(resp, "/api/track (phản hồi không phải JSON)")
    job_id = payload["job_id"]
    if payload.get("video_md5") and payload["video_md5"] != local_md5:
        raise RuntimeError(
            "Colab nhận video khác file local — upload lại."
        )
    print(f"[colab] remote job_id={job_id}", flush=True)
    t0 = time.monotonic()

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
            elapsed = time.monotonic() - t0
            if local_size > 5_000_000 and elapsed < _MIN_TRACKING_SEC:
                raise RuntimeError(
                    f"Colab báo xong sau {elapsed:.0f}s — có thể chưa chạy GPU thật "
                    f"(cần > {_MIN_TRACKING_SEC:.0f}s). Xem log cell Colab."
                )
            if data.get("video_md5") and data["video_md5"] != local_md5:
                raise RuntimeError("Colab tracking xong nhưng MD5 video không khớp.")
            print(
                f"[colab] tracking done in {elapsed:.0f}s "
                f"remote_md5={data.get('video_md5')}",
                flush=True,
            )
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

    if not stub_matches_video(dest_stub_path, video_path):
        os.remove(dest_stub_path)
        raise RuntimeError(
            "Stub từ Colab không khớp video vừa upload — Colab có thể dùng file cũ. "
            "Trên Colab: git pull → chạy lại cell server."
        )

    _notify("Đã nhận stub từ Colab (GPU).")
    print(f"[colab] stub OK md5_video={local_md5}", flush=True)
    return dest_stub_path

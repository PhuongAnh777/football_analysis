#!/usr/bin/env python3
"""Verify track_stubs.pkl is a valid binary pickle (not a truncated/HTML download)."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.stub_io import load_track_stub


def verify_stub(path: str) -> int:
    if not os.path.exists(path):
        print(f"FAIL: file not found: {path}")
        return 1

    size = os.path.getsize(path)
    print(f"Path : {path}")
    print(f"Size : {size:,} bytes ({size / 1024**2:.2f} MB)")

    with open(path, "rb") as fh:
        head = fh.read(64)
    print(f"Head : {head[:16]!r}")

    if head.lstrip().startswith((b"<!DOCTYPE", b"<html", b"<HTML")):
        print("FAIL: looks like HTML (Drive/Colab error page), not a pickle file.")
        return 1

    if not head.startswith((b"\x80\x03", b"\x80\x04", b"\x80\x05")):
        print("FAIL: missing pickle magic bytes (\\x80\\x0x).")
        return 1

    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    print(f"MD5  : {md5}")

    try:
        tracks, fps, enriched = load_track_stub(path)
    except Exception as exc:
        print(f"FAIL: pickle load error: {exc}")
        return 1

    n_frames = len(tracks.get("players", []))
    print(f"OK   : {n_frames} frames, fps={fps}, enriched={enriched}")

    if size < 500_000:
        print("WARN : file very small — likely truncated (expect several MB for full video).")
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify track_stubs.pkl")
    parser.add_argument(
        "path",
        nargs="?",
        default=os.path.join(_PROJECT_ROOT, "stubs", "track_stubs.pkl"),
    )
    args = parser.parse_args()
    raise SystemExit(verify_stub(args.path))


if __name__ == "__main__":
    main()

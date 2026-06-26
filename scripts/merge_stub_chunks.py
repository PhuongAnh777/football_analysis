#!/usr/bin/env python3
"""Merge stub_chunk_*.bin files downloaded from Colab into stubs/track_stubs.pkl."""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import shutil
import sys
import zipfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_chunk_dir(path: str) -> str:
    """Resolve chunk folder; ``Downloads`` → user Downloads on Windows."""
    if os.path.isdir(path):
        return os.path.abspath(path)

    # Common mistake: --chunk-dir Downloads (relative to cwd, not user home)
    user_downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    if path.replace("\\", "/").rstrip("/").endswith("Downloads") and os.path.isdir(user_downloads):
        print(f"Using user Downloads: {user_downloads}")
        return user_downloads

    return os.path.abspath(path)


def read_manifest(chunk_dir: str) -> dict[str, str]:
    manifest_path = os.path.join(chunk_dir, "manifest.txt")
    if not os.path.exists(manifest_path):
        return {}

    data: dict[str, str] = {}
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip()
    return data


def find_chunk_paths(chunk_dir: str) -> list[str]:
    pattern = os.path.join(chunk_dir, "stub_chunk_*.bin")
    paths = glob.glob(pattern)
    return sorted(paths, key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))


def merge_chunks(chunk_dir: str, output: str, expected_md5: str | None = None) -> None:
    chunk_dir = resolve_chunk_dir(chunk_dir)
    print(f"Chunk dir: {chunk_dir}")

    manifest = read_manifest(chunk_dir)
    expected_chunks = int(manifest["chunks"]) if "chunks" in manifest else None
    if not expected_md5 and "md5" in manifest:
        expected_md5 = manifest["md5"]
        print(f"MD5 from manifest: {expected_md5}")

    paths = find_chunk_paths(chunk_dir)
    if not paths:
        raise FileNotFoundError(
            f"No stub_chunk_*.bin in {chunk_dir}\n"
            f"Tip: use full path, e.g. --chunk-dir \"{os.path.join(os.path.expanduser('~'), 'Downloads')}\""
        )

    found_indices = [int(re.search(r"(\d+)", os.path.basename(p)).group(1)) for p in paths]
    print(f"Found {len(paths)} chunk(s): {found_indices[0]:03d}..{found_indices[-1]:03d}")

    if expected_chunks is not None:
        expected_indices = list(range(expected_chunks))
        missing = [i for i in expected_indices if i not in found_indices]
        if missing:
            missing_str = ", ".join(f"{i:03d}" for i in missing[:10])
            if len(missing) > 10:
                missing_str += f", ... (+{len(missing) - 10} more)"
            raise FileNotFoundError(
                f"Missing {len(missing)}/{expected_chunks} chunks: {missing_str}\n"
                "Download remaining stub_chunk_*.bin from Colab cell 7, then run again."
            )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    total = 0
    with open(output, "wb") as out:
        for path in paths:
            with open(path, "rb") as src:
                data = src.read()
            out.write(data)
            total += len(data)
            print(f"  + {os.path.basename(path)} ({len(data):,} bytes)")

    print(f"Wrote {output} ({total:,} bytes = {total / 1024**2:.2f} MB)")

    if "size_bytes" in manifest:
        expected_size = int(manifest["size_bytes"])
        if total != expected_size:
            raise ValueError(f"Size mismatch: got {total:,}, expected {expected_size:,}")

    md5 = hashlib.md5(open(output, "rb").read()).hexdigest()
    print(f"MD5: {md5}")
    if expected_md5 and md5.lower() != expected_md5.lower():
        raise ValueError(f"MD5 mismatch: got {md5}, expected {expected_md5}")


def extract_zip(zip_path: str) -> str:
    """Extract stub_chunks.zip to a sibling folder; return chunk dir path."""
    zip_path = os.path.abspath(zip_path)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    out_dir = os.path.splitext(zip_path)[0]
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    print(f"Extracted: {zip_path} → {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Colab stub chunks")
    parser.add_argument(
        "--chunk-dir",
        default=None,
        help="Folder with stub_chunk_*.bin (default: ~/Downloads or extracted ZIP)",
    )
    parser.add_argument(
        "--zip",
        default=None,
        help="Path to stub_chunks.zip from Colab (auto-extracts before merge)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_PROJECT_ROOT, "stubs", "track_stubs.pkl"),
    )
    parser.add_argument("--md5", default=None, help="Expected MD5 from Colab (optional)")
    args = parser.parse_args()

    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    chunk_dir = args.chunk_dir

    if args.zip:
        chunk_dir = extract_zip(args.zip)
    elif chunk_dir is None:
        zip_default = os.path.join(downloads, "stub_chunks.zip")
        extracted = os.path.join(downloads, "stub_chunks")
        if os.path.isdir(extracted) and find_chunk_paths(extracted):
            chunk_dir = extracted
        elif os.path.exists(zip_default):
            chunk_dir = extract_zip(zip_default)
        elif find_chunk_paths(downloads):
            chunk_dir = downloads
        else:
            chunk_dir = downloads

    merge_chunks(chunk_dir, args.output, args.md5)
    print("Run: python scripts/verify_stub.py", args.output)


if __name__ == "__main__":
    main()

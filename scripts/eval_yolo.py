#!/usr/bin/env python3
"""
Evaluate YOLO detection weights on the football-players-detection dataset.

Reports mAP@0.5, mAP@0.5:0.95, mean precision, mean recall, and per-class AP.

Usage:
    python scripts/eval_yolo.py
    python scripts/eval_yolo.py --split test --conf 0.25
    python scripts/eval_yolo.py --model models/best.pt --output eval_results/yolo_val.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DEFAULT_MODEL = os.path.join(_PROJECT_ROOT, "models", "best.pt")
_DATASET_ROOT  = os.path.join(_PROJECT_ROOT, "training", "football-players-detection-1")
_ORIGINAL_YAML = os.path.join(_DATASET_ROOT, "data.yaml")

_CLASS_NAMES = ("ball", "goalkeeper", "player", "referee")


def _dataset_dir() -> Path:
    """Return the directory that actually contains train/valid/test splits."""
    nested = Path(_DATASET_ROOT) / "football-players-detection-1"
    if (nested / "valid" / "images").is_dir():
        return nested
    flat = Path(_DATASET_ROOT)
    if (flat / "valid" / "images").is_dir():
        return flat
    raise FileNotFoundError(
        "Could not find dataset images. Expected one of:\n"
        f"  {_DATASET_ROOT}/football-players-detection-1/valid/images\n"
        f"  {_DATASET_ROOT}/valid/images"
    )


def resolve_data_yaml() -> str:
    """
    Build a data.yaml with paths relative to the dataset root.

    The bundled Roboflow export nests splits one level deeper than the
    original data.yaml declares; this helper writes a corrected YAML.
    """
    root = _dataset_dir()
    lines = [
        f"path: {root.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        f"nc: {len(_CLASS_NAMES)}",
        "names:",
    ]
    lines.extend(f"  - {name}" for name in _CLASS_NAMES)

    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="yolo_eval_")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _count_images(split_dir: Path) -> int:
    if not split_dir.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in split_dir.iterdir() if p.suffix.lower() in exts)


def run_eval(
    *,
    model_path: str = _DEFAULT_MODEL,
    data_yaml: str | None = None,
    split: str = "val",
    imgsz: int = 640,
    conf: float | None = None,
    iou: float = 0.7,
    device: str | None = None,
    project: str = "runs/detect",
    name: str = "eval",
    output_json: str | None = None,
) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    from ultralytics import YOLO

    temp_yaml: str | None = None
    if data_yaml is None:
        temp_yaml = resolve_data_yaml()
        data_yaml = temp_yaml
    elif not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    root = _dataset_dir()
    split_dir = root / {"train": "train", "val": "valid", "test": "test"}[split] / "images"
    n_images = _count_images(split_dir)
    if n_images == 0:
        raise FileNotFoundError(f"No images in split '{split}': {split_dir}")

    print(f"[eval] Model : {model_path}")
    print(f"[eval] Data  : {data_yaml}")
    print(f"[eval] Split : {split} ({n_images} images)")
    print(f"[eval] imgsz : {imgsz}")

    model = YOLO(model_path)
    val_kwargs: dict = {
        "data":    data_yaml,
        "split":   split,
        "imgsz":   imgsz,
        "iou":     iou,
        "project": os.path.join(_PROJECT_ROOT, project),
        "name":    name,
        "verbose": False,
    }
    if conf is not None:
        val_kwargs["conf"] = conf
        print(f"[eval] conf  : {conf}")
    if device is not None:
        val_kwargs["device"] = device

    metrics = model.val(**val_kwargs)

    names = getattr(metrics, "names", None) or dict(enumerate(_CLASS_NAMES))
    per_class = {}
    raw_maps = getattr(metrics.box, "maps", None)
    maps = list(raw_maps) if raw_maps is not None else []
    for cls_id, cls_name in names.items():
        idx = int(cls_id)
        per_class[cls_name] = {
            "ap50_95": round(float(maps[idx]), 4) if idx < len(maps) else None,
        }

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model":     os.path.relpath(model_path, _PROJECT_ROOT),
        "data_yaml": data_yaml,
        "split":     split,
        "n_images":  n_images,
        "imgsz":     imgsz,
        "conf":      conf,
        "iou":       iou,
        "metrics": {
            "map50_95": round(float(metrics.box.map),    4),
            "map50":    round(float(metrics.box.map50),  4),
            "precision": round(float(metrics.box.mp),     4),
            "recall":    round(float(metrics.box.mr),     4),
        },
        "per_class": per_class,
        "save_dir":  str(getattr(metrics, "save_dir", "")),
    }

    if temp_yaml and os.path.exists(temp_yaml):
        os.remove(temp_yaml)

    if output_json:
        out_path = output_json if os.path.isabs(output_json) else os.path.join(_PROJECT_ROOT, output_json)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"[eval] JSON  : {out_path}")

    return result


def _print_summary(result: dict) -> None:
    m = result["metrics"]
    print()
    print("=" * 52)
    print("YOLO Detection Evaluation")
    print("=" * 52)
    print(f"  mAP@0.5:0.95 : {m['map50_95']:.4f}")
    print(f"  mAP@0.5      : {m['map50']:.4f}")
    print(f"  Precision    : {m['precision']:.4f}")
    print(f"  Recall       : {m['recall']:.4f}")
    print("-" * 52)
    print("  Per-class AP@0.5:0.95:")
    for cls_name, vals in result["per_class"].items():
        ap = vals.get("ap50_95")
        ap_str = f"{ap:.4f}" if ap is not None else "n/a"
        print(f"    {cls_name:12s} : {ap_str}")
    print("=" * 52)
    if result.get("save_dir"):
        print(f"  Plots saved → {result['save_dir']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO detection (mAP, precision, recall)"
    )
    parser.add_argument(
        "--model", default=_DEFAULT_MODEL,
        help="Path to .pt weights (default: models/best.pt)",
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to data.yaml (default: auto-resolve from training/)",
    )
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="val",
        help="Dataset split to evaluate (default: val)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size")
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Confidence threshold (default: Ultralytics val default ~0.001)",
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", default=None, help="cuda, cpu, or device id")
    parser.add_argument(
        "--project", default="runs/detect",
        help="Ultralytics project dir for plots",
    )
    parser.add_argument("--name", default="eval", help="Run name under project/")
    parser.add_argument(
        "--output", default="eval_results/yolo_eval.json",
        help="Path to save JSON summary (default: eval_results/yolo_eval.json)",
    )
    parser.add_argument(
        "--no-json", action="store_true",
        help="Skip writing JSON output file",
    )
    args = parser.parse_args()

    os.chdir(_PROJECT_ROOT)
    result = run_eval(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        output_json=None if args.no_json else args.output,
    )
    _print_summary(result)


if __name__ == "__main__":
    main()

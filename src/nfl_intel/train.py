"""Fine-tuning helpers: sample + pre-label keyframes, write the dataset YAML, train YOLO.

Workflow
  1. `nfl prelabel data/clips/*.mp4 --every 30`  → data/labels/images/*.jpg + labels/*.txt
     (YOLO format, pre-filled by the current detector so you only correct boxes)
  2. Fix the boxes in X-AnyLabeling (or any tool that reads YOLO txt), classes: player, referee
  3. `nfl train --epochs 40`                        → runs/detect/nfl/weights/best.pt
  4. `nfl overlay --weights runs/detect/nfl/weights/best.pt ...`
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["player", "referee"]


def yolo_line(box, width: int, height: int, cls: int = 0) -> str:
    """xyxy pixels -> 'cls cx cy w h' normalised, the YOLO label format."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    w, h = (x2 - x1) / width, (y2 - y1) / height
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def write_dataset_yaml(root: Path, val_fraction: float = 0.2, seed: int = 0) -> Path:
    """Split images into train/val lists and write data.yaml for Ultralytics."""
    images = sorted((root / "images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no images under {root / 'images'}; run `nfl prelabel` first")
    rng = random.Random(seed)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * val_fraction))
    val, train = images[:n_val], images[n_val:]
    (root / "train.txt").write_text("\n".join(str(p.resolve()) for p in train) + "\n")
    (root / "val.txt").write_text("\n".join(str(p.resolve()) for p in val) + "\n")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root.resolve()}\ntrain: train.txt\nval: val.txt\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    )
    return yaml_path


def prelabel(videos: list[str], out_dir: str | Path, detector, every: int = 30, max_per_clip: int | None = None) -> int:
    """Save every Nth frame as a JPEG with the detector's boxes as YOLO labels. Returns frames written."""
    from .capture import VideoSource

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    written = 0
    for path in videos:
        src = VideoSource(path)
        stem = Path(path).stem
        n = 0
        for idx, _t, frame in src:
            if idx % every:
                continue
            if max_per_clip is not None and n >= max_per_clip:
                break
            h, w = frame.shape[:2]
            dets = detector(frame)
            name = f"{stem}_{idx:05d}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines = [yolo_line(b, w, h, 0) for b in np.asarray(dets.xyxy).reshape(-1, 4)]
            (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
            written += 1
        src.release()
    return written


def fine_tune(
    data_yaml: str | Path,
    weights: str = "yolo11s.pt",
    epochs: int = 40,
    imgsz: int = 1280,
    batch: int = 4,
    freeze_epochs: int = 10,
    device: str | None = None,
    name: str = "nfl",
) -> Path:
    """Two-stage fine-tune: frozen backbone first, then everything, small-object friendly.

    Recipe from the Ultralytics fine-tuning guide and what worked on football MOT sets:
    imgsz 1280, AdamW, low LR, mosaic on, no rotation/shear (the field is always level).
    Returns the path to best.pt.
    """
    from ultralytics import YOLO

    from .detect import pick_device

    device = device or pick_device()
    common = dict(
        data=str(data_yaml), imgsz=imgsz, batch=batch, device=device, optimizer="AdamW",
        lr0=0.001, warmup_epochs=1, mosaic=1.0, close_mosaic=10, scale=0.5, fliplr=0.5,
        degrees=0.0, shear=0.0, patience=15, plots=True, verbose=False,
    )
    model = YOLO(weights)
    if freeze_epochs > 0:
        model.train(epochs=freeze_epochs, freeze=10, name=f"{name}-stage1", **common)
        model = YOLO(Path(model.trainer.save_dir) / "weights" / "last.pt")
    model.train(epochs=epochs, name=name, **common)
    return Path(model.trainer.save_dir) / "weights" / "best.pt"

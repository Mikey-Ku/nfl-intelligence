"""Player detection with a pretrained YOLO model.

Out of the box this uses COCO weights and keeps the `person` class, which already finds
players on film. Fine-tuning on labeled frames (player / referee / ball) swaps in through
`weights=` with no other change.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

DEFAULT_WEIGHTS = "yolo11n.pt"


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PlayerDetector:
    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        device: str | None = None,
        conf: float = 0.3,
        imgsz: int = 1280,
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device or pick_device()
        self.conf = conf
        self.imgsz = imgsz
        names = self.model.names
        # COCO weights: keep only `person`. Custom weights: keep every class.
        self.classes = [i for i, n in names.items() if n == "person"] or None

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)

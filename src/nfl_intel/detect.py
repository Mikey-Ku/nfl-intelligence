"""Player detection with a pretrained YOLO model.

Out of the box this uses COCO weights and keeps the `person` class, which already finds
players on film. Weights fine-tuned on the corrected frames (player / referee) swap in
through `weights=`; then only `player` is kept, so the officials drop out here.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

DEFAULT_WEIGHTS = "yolo11s.pt"


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def keep_classes(names: dict[int, str]) -> list[int] | None:
    """The classes that count as players: `person` on COCO weights, `player` on weights
    fine-tuned here (which also know `referee`, dropped so officials are never tracked).
    Any other custom weights keep every class."""
    for wanted in ("player", "person"):
        found = [i for i, n in names.items() if n == wanted]
        if found:
            return found
    return None


class PlayerDetector:
    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        device: str | None = None,
        conf: float = 0.3,
        imgsz: int = 1280,
        tiles: int = 1,
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device or pick_device()
        self.conf = conf
        self.imgsz = imgsz
        self.tiles = max(1, tiles)
        names = self.model.names
        self.classes = keep_classes(names)
        self._slicer: sv.InferenceSlicer | None = None
        self._slicer_size: tuple[int, int] | None = None

    def _predict(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            max_det=60,                       # a crowd shot can hold hundreds of people; a play never does
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        """Whole-frame inference, or `tiles` x `tiles` overlapping slices merged with NMS.

        Tiling costs ~tiles² inference passes but finds the small far-sideline players
        that a single 1280-px pass misses on a 4K grab.
        """
        if self.tiles == 1:
            return self._predict(frame)
        h, w = frame.shape[:2]
        size = (w // self.tiles, h // self.tiles)
        if self._slicer is None or self._slicer_size != size:
            self._slicer = sv.InferenceSlicer(
                callback=self._predict,
                slice_wh=size,
                overlap_wh=(size[0] // 5, size[1] // 5),
                iou_threshold=0.5,
                thread_workers=1,
            )
            self._slicer_size = size
        return self._slicer(frame)

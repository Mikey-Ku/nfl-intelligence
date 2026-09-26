"""Field model in yards, pixel↔field homography, and field drawing.

Coordinates follow the Big Data Bowl convention: x runs the length of the field from the
back of the left end zone (0) to the back of the right end zone (120), so the goal lines are
at x=10 and x=110 and midfield is x=60. y runs across the field from the near sideline (0)
to the far sideline (53.3).

Landmarks are named `<column>-<row>` so a calibration prompt can say "click L40-H1":
  columns  LEZ (x=0), L0..L50 (left goal line and yard lines, x=10..60), 50 (x=60),
           R50..R0 (x=60..110), REZ (x=120)
  rows     S1 near sideline, H1 near hash, H2 far hash, S2 far sideline
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

FIELD_LENGTH = 120.0
FIELD_WIDTH = 160.0 / 3.0           # 53.33 yards
HASH_INSET = 70.75 / 3.0            # NFL hashes sit 70' 9" from each sideline
ROWS = {
    "S1": 0.0,
    "H1": HASH_INSET,
    "H2": FIELD_WIDTH - HASH_INSET,
    "S2": FIELD_WIDTH,
}


def column_x(col: str) -> float:
    col = col.upper().strip()
    if col == "LEZ":
        return 0.0
    if col == "REZ":
        return FIELD_LENGTH
    if col == "50":
        return 60.0
    side, num = col[0], col[1:]
    if side not in "LR" or not num.isdigit():
        raise ValueError(f"bad column {col!r}; expected LEZ, L0..L50, 50, R50..R0, REZ")
    n = int(num)
    if not 0 <= n <= 50:
        raise ValueError(f"yard number out of range in {col!r}")
    return 10.0 + n if side == "L" else 110.0 - n


def landmark_xy(name: str) -> tuple[float, float]:
    """'L40-H1' -> (50.0, 23.58)."""
    try:
        col, row = name.upper().strip().split("-")
    except ValueError as e:
        raise ValueError(f"bad landmark {name!r}; expected e.g. L40-H1") from e
    if row not in ROWS:
        raise ValueError(f"bad row in {name!r}; expected one of {list(ROWS)}")
    return column_x(col), ROWS[row]


class FieldCalibration:
    """Maps source pixels to field yards through a 3x3 homography."""

    def __init__(self, H: np.ndarray, meta: dict | None = None):
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError("homography must be 3x3")
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.meta = meta or {}

    @classmethod
    def from_correspondences(cls, px, yd, meta: dict | None = None) -> FieldCalibration:
        px = np.asarray(px, dtype=np.float32).reshape(-1, 1, 2)
        yd = np.asarray(yd, dtype=np.float32).reshape(-1, 1, 2)
        if len(px) < 4:
            raise ValueError("need at least 4 point pairs")
        method = 0 if len(px) == 4 else cv2.RANSAC
        H, _ = cv2.findHomography(px, yd, method, 1.0)
        if H is None:
            raise ValueError("could not fit a homography (points collinear?)")
        return cls(H, meta)

    @classmethod
    def from_landmarks(cls, px, names, meta: dict | None = None) -> FieldCalibration:
        return cls.from_correspondences(px, [landmark_xy(n) for n in names], meta)

    def to_field(self, pts_px) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_pixels(self, pts_yd) -> np.ndarray:
        pts = np.asarray(pts_yd, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def reprojection_error(self, px, yd) -> float:
        """Mean distance in yards between mapped pixels and their true field points."""
        pred = self.to_field(px)
        return float(np.linalg.norm(pred - np.asarray(yd, dtype=np.float64), axis=1).mean())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"H": self.H.tolist(), "meta": self.meta}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> FieldCalibration:
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["H"]), d.get("meta"))


# ---------------------------------------------------------------- drawing

GRASS = (60, 120, 40)
LINE = (235, 235, 235)


def draw_field(scale: float = 8.0, pad: float = 2.0) -> np.ndarray:
    """Top-down field as a BGR image; `scale` is pixels per yard, `pad` is yards of margin."""
    w = int(round((FIELD_LENGTH + 2 * pad) * scale))
    h = int(round((FIELD_WIDTH + 2 * pad) * scale))
    img = np.full((h, w, 3), GRASS, dtype=np.uint8)

    def P(x, y):
        return int(round((x + pad) * scale)), int(round((y + pad) * scale))

    # end zones
    for x0 in (0.0, 110.0):
        cv2.rectangle(img, P(x0, 0), P(x0 + 10, FIELD_WIDTH), (45, 95, 30), -1)
    # yard lines every 5 yards, thicker at the goal lines
    for x in range(0, 121, 5):
        thick = 2 if x in (10, 110) else 1
        cv2.line(img, P(x, 0), P(x, FIELD_WIDTH), LINE, thick)
    # hash ticks every yard
    tick = 0.5
    for x in range(11, 110):
        for y in (0.0, HASH_INSET, FIELD_WIDTH - HASH_INSET, FIELD_WIDTH):
            y0 = y - tick if y > 1 else y
            cv2.line(img, P(x, y0), P(x, y0 + tick), LINE, 1)
    # sidelines
    cv2.rectangle(img, P(0, 0), P(FIELD_LENGTH, FIELD_WIDTH), LINE, 2)
    # numbers every 10 yards
    for x in range(20, 101, 10):
        n = x - 10 if x <= 60 else 110 - x
        label = str(n)
        for y in (9.0, FIELD_WIDTH - 9.0):
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale / 8, 1)
            cx, cy = P(x, y)
            cv2.putText(img, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35 * scale / 8, LINE, 1, cv2.LINE_AA)
    return img


def field_to_image(pts_yd, scale: float = 8.0, pad: float = 2.0) -> np.ndarray:
    """Field yards -> pixel coordinates on an image from draw_field(scale, pad)."""
    pts = np.asarray(pts_yd, dtype=np.float64).reshape(-1, 2)
    return (pts + pad) * scale

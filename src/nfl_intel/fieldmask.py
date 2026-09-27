"""Where the grass is: a field mask from the frame's green, its bounding box, and its colour.

Used three ways: the detector runs on a crop around the field instead of the whole browser
window (2-3x more pixels per player), people whose feet aren't on the turf are dropped
(bench, photographers, crowd), and the grass colour is what the jersey-colour feature masks
out. `grass_mask` is only the loose first pass; `find_field` does the real work.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FieldInfo:
    mask: np.ndarray            # full-resolution bool mask of the playing surface (holes filled)
    grass: np.ndarray           # full-resolution bool mask of grass-coloured pixels (raw)
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 of the field in pixels
    band: tuple[int, int]       # first and last row of the turf band in pixels
    grass_lab: np.ndarray       # median Lab colour of the grass
    fraction: float             # share of the frame that is field

    def contains_foot(self, x: float, y: float) -> bool:
        h, w = self.mask.shape
        xi, yi = int(round(x)), int(round(y))
        return 0 <= xi < w and 0 <= yi < h and bool(self.mask[yi, xi])

    far_sideline: tuple[float, float] | None = None   # y = a*x + b in pixels, or None if not found
    near_sideline: tuple[float, float] | None = None

    def foot_on_field(self, x: float, y: float, margin_frac: float = 0.015) -> bool:
        """Is a player standing on the field?

        Inside the turf band, and between the sidelines when they were found. The turf runs
        on past the sideline paint (that's where the benches stand), so colour alone can't
        tell a wide receiver on the far sideline from a coach two yards behind it; the
        painted line can. Feet on the line itself count as on the field.
        """
        if not self.contains_foot(x, y):
            return False
        margin = margin_frac * (self.band[1] - self.band[0])
        if self.far_sideline is not None:
            a, b = self.far_sideline
            if y < a * x + b - margin:
                return False
        if self.near_sideline is not None:
            a, b = self.near_sideline
            if y > a * x + b + margin:
                return False
        return True


def grass_mask(frame: np.ndarray) -> np.ndarray:
    """Green, reasonably saturated pixels (OpenCV hue 35-90 covers turf under lights)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0] >= 35) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 45) & (hsv[..., 2] >= 40)


def turf_rows(mask: np.ndarray, max_gap: int, rel_threshold: float = 0.25, min_run: int = 3) -> tuple[int, int] | None:
    """Longest band of rows whose grass fraction is at least `rel_threshold` of the best rows.

    A row only counts if it starts a run of `min_run` consecutive grass rows (a stray green
    row in the crowd doesn't), and gaps up to `max_gap` rows are bridged (end-zone
    lettering, the far sideline paint, a pile-up).
    """
    frac = mask.mean(axis=1)
    top = float(np.percentile(frac, 90))
    if top <= 0:
        return None
    ok = frac > rel_threshold * top
    # keep only rows that belong to a run of at least min_run consecutive ok rows
    solid = np.zeros_like(ok)
    r = 0
    while r < len(ok):
        if ok[r]:
            s = r
            while r < len(ok) and ok[r]:
                r += 1
            if r - s >= min_run:
                solid[s:r] = True
        else:
            r += 1
    best, start, last_ok = None, None, None
    for r, v in enumerate(solid):
        if v:
            if start is None or (last_ok is not None and r - last_ok - 1 > max_gap):
                start = r
            last_ok = r
            if best is None or last_ok - start > best[1] - best[0]:
                best = (start, last_ok)
    return best


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes: players, numbers and logos punch holes in the grass, but feet
    standing inside the field's outline are still on the field."""
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = mask
    outside = (1 - padded).copy()
    cv2.floodFill(outside, None, (0, 0), 2)        # background connected to the border -> 2
    padded[outside == 1] = 1                        # background not connected to the border = hole
    return padded[1:-1, 1:-1]


def find_field(
    frame: np.ndarray,
    scale: float = 0.125,
    min_fraction: float = 0.05,
    dilate_frac: float = 0.015,
    grass_tolerance: float = 30.0,
) -> FieldInfo | None:
    """The turf: pixels close in colour to the grass, largest blob, holes filled, dilated.

    Two passes: a loose hue mask finds green, its median Lab colour in the middle of the
    frame is taken as "the grass", and the final mask keeps only pixels within
    `grass_tolerance` of it. That separates turf from green stadium walls, green scoreboard
    panels and dark-green jerseys, which a plain hue mask lumps together.
    Returns None when less than `min_fraction` of the frame is grass (menus, replays, ads).
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float64)
    green = grass_mask(small)
    centre = np.zeros_like(green)
    centre[sh // 4 : 3 * sh // 4, sw // 4 : 3 * sw // 4] = True
    sample = lab[green & centre] if (green & centre).sum() >= 20 else lab[green]
    if len(sample) < 20:
        return None
    grass_lab = np.median(sample, axis=0)
    raw = (np.linalg.norm(lab - grass_lab, axis=2) < grass_tolerance).astype(np.uint8)
    # Keep only the band of rows that is mostly turf. A green-clad crowd and a green stadium
    # wall have some grass-coloured pixels, but never row after row of them.
    band = turf_rows(raw, max_gap=int(sh * 0.12))
    if band is None:
        return None
    mask = raw.copy()
    mask[: band[0]] = 0
    mask[band[1] + 1 :] = 0
    k = max(3, int(sh * 0.06) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    field = fill_holes((labels == biggest).astype(np.uint8))
    fraction = float(field.mean())
    if fraction < min_fraction:
        return None
    d = max(1, int(sh * dilate_frac) | 1)
    field = cv2.dilate(field, np.ones((d, d), np.uint8))
    x, y, bw, bh = cv2.boundingRect(field)
    full = cv2.resize(field, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    grass_full = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    bbox = (int(x / scale), int(y / scale), min(w, int((x + bw) / scale)), min(h, int((y + bh) / scale)))
    band_px = (int(band[0] / scale), min(h - 1, int((band[1] + 1) / scale)))
    info = FieldInfo(mask=full, grass=grass_full, bbox=bbox, band=band_px, grass_lab=grass_lab, fraction=fraction)
    info.far_sideline = find_sideline(frame, info, "far")
    info.near_sideline = find_sideline(frame, info, "near")
    return info


def white_mask(bgr: np.ndarray) -> np.ndarray:
    """Bright, colourless pixels: painted lines."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return (lab[..., 0] >= 170) & (np.abs(lab[..., 1].astype(int) - 128) < 14) & (np.abs(lab[..., 2].astype(int) - 128) < 18)


def find_sideline(frame: np.ndarray, field: FieldInfo, which: str, scale: float = 0.25, edge_frac: float = 0.35) -> tuple[float, float] | None:
    """The painted sideline as y = a*x + b (full-res pixels), or None.

    Looks for the longest near-horizontal white line in the top (`far`) or bottom (`near`)
    part of the turf band. Yard lines run the other way, hash marks are short, and end-zone
    lettering is neither white enough nor long enough to win.
    """
    x1, _, x2, _ = field.bbox
    top, bottom = field.band
    span = bottom - top
    if which == "far":
        y1, y2 = max(0, top - int(0.05 * span)), top + int(edge_frac * span)
    else:
        y1, y2 = bottom - int(edge_frac * span), min(frame.shape[0], bottom + int(0.05 * span))
    if y2 - y1 < 8 or x2 - x1 < 8:
        return None
    roi = frame[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(8, int(roi.shape[1] * scale)), max(8, int(roi.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    mask = white_mask(small).astype(np.uint8) * 255
    min_len = int(small.shape[1] * 0.3)
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=max(20, min_len // 3), minLineLength=min_len, maxLineGap=int(small.shape[1] * 0.03))
    if lines is None:
        return None
    best = None
    for lx1, ly1, lx2, ly2 in np.asarray(lines).reshape(-1, 4):
        dx, dy = lx2 - lx1, ly2 - ly1
        if dx == 0 or abs(dy / dx) > 0.25:          # steeper than ~14 degrees: not a sideline
            continue
        length = float(np.hypot(dx, dy))
        if best is None or length > best[0]:
            best = (length, lx1, ly1, lx2, ly2)
    if best is None:
        return None
    _, lx1, ly1, lx2, ly2 = best
    fx1, fy1 = x1 + lx1 / scale, y1 + ly1 / scale
    fx2, fy2 = x1 + lx2 / scale, y1 + ly2 / scale
    a = (fy2 - fy1) / (fx2 - fx1)
    return float(a), float(fy1 - a * fx1)


def expand_bbox(bbox: tuple[int, int, int, int], frame_shape, margin: float = 0.08) -> tuple[int, int, int, int]:
    """Grow a box by `margin` of its size on each side, clamped to the frame."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)

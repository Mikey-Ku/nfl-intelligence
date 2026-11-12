"""Read the yard numbers painted on the field.

Through the mapping the field can be laid flat, and on the flat picture the big painted
numbers are ordinary text. Each one says which yard line it sits on, and its arrow points at
the nearer goal line, so one number is enough to turn the mapping's own yard count into real
yard lines. That works with nothing but the video: no page header, no play-by-play.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from .field import FIELD_WIDTH

ARROWS = "-–—<>«»‹›◄►◀▶→←•·~_"
NUMBER_ROWS = (9.0, 17.0)                  # yards in from a sideline where the numbers are painted


@dataclass
class PaintedNumber:
    value: int                   # 10, 20, 30, 40 or 50
    x: float                     # the yard line it sits on, in the mapping's yards
    y: float
    goal: int                    # +1 / -1: which way along x the nearer goal line lies; 0 when unknown (or at the 50)
    confidence: float


@dataclass
class FlatField:
    img: np.ndarray
    scale: float
    lo: float
    hi: float
    mirrored: bool

    def field_x(self, px: float) -> float:
        return self.hi - px / self.scale if self.mirrored else self.lo + px / self.scale


def flatten(frame: np.ndarray, H: np.ndarray, rect, boxes=None, scale: float = 24.0, grass=(60, 120, 60)) -> FlatField | None:
    """The part of the field in view, seen from straight above, never as a mirror image
    (mirrored digits can't be read). Players and everything outside the video are painted out."""
    x1, y1, x2, y2 = rect
    probe = np.array([[x1, y2], [x2, y2], [x1, (y1 + y2) / 2], [x2, (y1 + y2) / 2]], dtype=np.float64).reshape(-1, 1, 2)
    corners = cv2.perspectiveTransform(probe, H).reshape(-1, 2)
    if not np.all(np.isfinite(corners)):
        return None
    lo, hi = float(np.floor(corners[:, 0].min())), float(np.ceil(corners[:, 0].max()))
    if not 10.0 <= hi - lo <= 110.0:
        return None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    q = cv2.perspectiveTransform(np.array([[cx, cy], [cx + 50, cy], [cx, cy + 50]], dtype=np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)
    a, b = q[1] - q[0], q[2] - q[0]
    mirrored = a[0] * b[1] - a[1] * b[0] < 0
    if mirrored:
        to_img = np.array([[-scale, 0.0, hi * scale], [0.0, scale, 0.0], [0.0, 0.0, 1.0]])
    else:
        to_img = np.array([[scale, 0.0, -lo * scale], [0.0, scale, 0.0], [0.0, 0.0, 1.0]])
    size = (int((hi - lo) * scale), int(FIELD_WIDTH * scale))
    seen = np.zeros(frame.shape[:2], np.uint8)
    seen[y1:y2, x1:x2] = 255
    if boxes is not None:
        for bx1, by1, bx2, by2 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
            seen[max(0, int(by1) - 4) : int(by2) + 4, max(0, int(bx1) - 4) : int(bx2) + 4] = 0
    M = to_img @ H
    flat = cv2.warpPerspective(frame, M, size, flags=cv2.INTER_LINEAR)
    flat[cv2.warpPerspective(seen, M, size, flags=cv2.INTER_NEAREST) == 0] = grass
    return FlatField(flat, scale, lo, hi, mirrored)


def parse_number(text: str) -> tuple[int, int] | None:
    """(yard number, where the arrow is: -1 before the digits, +1 after, 0 none) or None."""
    t = text.strip().replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1").replace("S", "5").replace(" ", "")
    m = re.fullmatch(rf"([{re.escape(ARROWS)}]*)([1-5])0([{re.escape(ARROWS)}]*)", t)
    if not m:
        return None
    before, after = bool(m.group(1)), bool(m.group(3))
    return int(m.group(2)) * 10, 0 if before == after else (-1 if before else 1)


def read_numbers(flat: FlatField, reader=None) -> list[PaintedNumber]:
    """Every painted number that can be read off the flat field."""
    if reader is None:
        from .ocr import read_text as reader
    out: list[PaintedNumber] = []
    for b in reader(flat.img):
        got = parse_number(b.text)
        if got is None or b.confidence < 0.5:
            continue
        value, arrow = got
        y = (b.box[1] + b.box[3]) / 2.0 / flat.scale
        from_side = min(y, FIELD_WIDTH - y)
        if not NUMBER_ROWS[0] <= from_side <= NUMBER_ROWS[1]:
            continue                                   # a number is painted in one of two rows; anything else is a logo or a jersey
        xa, xb = flat.field_x(b.box[0]), flat.field_x(b.box[2])
        width = abs(xb - xa)
        if not 2.5 <= width <= 7.5:
            continue
        # the text is written to be read from the nearer sideline: left to right is +x from the y = 0 side, -x from the other
        reads = 1 if y < FIELD_WIDTH / 2 else -1
        goal = 0 if value == 50 else arrow * reads
        centre = (xa + xb) / 2.0 - (goal * width / 6.0 if arrow else 0.0)      # the arrow is a third character: leave it out
        out.append(PaintedNumber(value, 5.0 * round(centre / 5.0), y, goal, float(b.confidence)))
    return out


def goal_line(numbers: list[PaintedNumber]) -> tuple[float, int] | None:
    """Where the nearer goal line is in the mapping's yards, and which way it lies (+1 / -1).

    Every number votes: with its arrow for one goal line, without it for either. A 50 has no
    arrow and no nearer goal line, but it stands fifty yards from both, so it backs whichever
    of another number's two candidates lies fifty yards from it: a 50 and a 30 twenty yards
    apart say everything. Two numbers that agree, or one with an arrow, settle it.
    """
    fifty = [n for n in numbers if n.value == 50]
    votes: dict[tuple[float, int], float] = {}
    for n in numbers:
        if n.value == 50:
            continue
        for g in ((n.goal,) if n.goal else (1, -1)):
            key = (n.x + g * n.value, g)
            votes[key] = votes.get(key, 0.0) + (1.0 if n.goal else 0.45) * n.confidence
    for (goal, side) in list(votes):
        near = [f for f in fifty if abs(f.x - (goal - side * 50.0)) <= 2.5]
        if near:
            votes[(goal, side)] += 0.6 * max(f.confidence for f in near)
    if not votes:
        return None if not fifty else (fifty[0].x + 50.0, 1)            # at midfield either goal line will do
    (goal, side), weight = max(votes.items(), key=lambda kv: kv[1])
    rival = max((w for (g, s), w in votes.items() if (g, s) != (goal, side)), default=0.0)
    if weight < 0.9 or weight - rival < 0.4:
        return None
    return float(goal), int(side)

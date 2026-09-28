"""Pin the relative field mapping to the real field.

Registration gives positions in yards, but measured from an arbitrary yard line and with no
idea which way the offense is going. The play-by-play says how far the line of scrimmage is
from the goal line being attacked. Put together with where the two teams face each other on
film, and with the end zone if it is in the picture, that fixes everything.

Absolute coordinates: the offense always moves toward +x and attacks the goal line at
x = 110, so the line of scrimmage is at x = 110 - yards_to_goal. Same frame as bdb.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .field import FIELD_WIDTH


@dataclass
class Anchor:
    los_rel: float               # line of scrimmage in the registration's own coordinates
    direction: int               # +1 if the offense moves toward +x in those coordinates
    los_abs: float               # 110 - yards to goal
    offense_team: int | None     # which jersey group (0 or 1) has the ball
    source: str                  # what settled the direction: "end zone" or "formation"
    split_accuracy: float        # share of players on their own side of the line

    def to_absolute(self, xy_rel) -> np.ndarray:
        xy = np.asarray(xy_rel, dtype=np.float64).reshape(-1, 2)
        x = self.los_abs + self.direction * (xy[:, 0] - self.los_rel)
        y = xy[:, 1] if self.direction > 0 else FIELD_WIDTH - xy[:, 1]     # a half turn, not a mirror
        return np.column_stack([x, y])

    def to_relative_x(self, x_abs: float) -> float:
        return self.los_rel + self.direction * (x_abs - self.los_abs)

    def yard_name(self, x_rel: float) -> str:
        """What is painted at this yard line: 10, 20, ... 50 ... 10, G at the goal lines."""
        x = self.los_abs + self.direction * (x_rel - self.los_rel)
        if abs(x - 10.0) < 0.01 or abs(x - 110.0) < 0.01:
            return "G"
        if x < 10.0 or x > 110.0:
            return ""
        return f"{x - 10.0 if x <= 60.0 else 110.0 - x:.0f}"

    def matrix(self) -> np.ndarray:
        """3x3 taking relative yards to absolute yards."""
        d = float(self.direction)
        return np.array([
            [d, 0.0, self.los_abs - d * self.los_rel],
            [0.0, d, 0.0 if d > 0 else FIELD_WIDTH],
            [0.0, 0.0, 1.0],
        ])


def split_teams(x: np.ndarray, team: np.ndarray) -> tuple[float, int, float] | None:
    """Where the two teams face each other.

    Before the snap each team stands on its own side of the line of scrimmage. Returns the x
    that best separates them, which team is on the low-x side, and the share of players that
    ends up on the right side.
    """
    x, team = np.asarray(x, dtype=np.float64), np.asarray(team)
    keep = (team == 0) | (team == 1)
    x, team = x[keep], team[keep]
    if (team == 0).sum() < 3 or (team == 1).sum() < 3:
        return None
    order = np.argsort(x)
    x, team = x[order], team[order]
    best = None
    for i in range(1, len(x)):
        cut = (x[i - 1] + x[i]) / 2.0
        gap = x[i] - x[i - 1]
        for low in (0, 1):
            right = int((team[:i] == low).sum() + (team[i:] != low).sum())
            key = (right, gap)                       # most players on their side; then the widest gap
            if best is None or key > best[0]:
                best = (key, cut, low)
    (right, _), cut, low = best
    return float(cut), int(low), right / len(x)


def find_end_zone(frame: np.ndarray, H: np.ndarray, rect, grass_lab: np.ndarray, boxes: np.ndarray | None = None,
                  min_inside: float = 0.12, max_outside: float = 0.05) -> tuple[float, int] | None:
    """The goal line and which way the end zone lies from it, in relative yards.

    End zones are decorated: lettering and colour that is neither grass nor white. Sample the
    field through the mapping, measure how decorated each yard-wide strip is, and look for a
    10-yard stretch of decoration next to 10 yards of plain grass, starting on a yard line.
    """
    x1, y1, x2, y2 = rect
    probe = np.array([[x1, y2], [x2, y2], [x1, (y1 + y2) / 2], [x2, (y1 + y2) / 2]], dtype=np.float64)
    corners = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), H).reshape(-1, 2)
    if not np.all(np.isfinite(corners)):
        return None
    lo, hi = np.floor(corners[:, 0].min()) - 5, np.ceil(corners[:, 0].max()) + 5
    xs = np.arange(lo, hi, 0.5) + 0.25                    # two looks inside every yard, none on the paint between yards
    ys = np.arange(3.0, FIELD_WIDTH - 3.0 + 0.01, 1.0)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    px = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    inside = np.all(np.isfinite(px), axis=1) & (px[:, 0] >= x1 + 2) & (px[:, 0] < x2 - 2) & (px[:, 1] >= y1 + 2) & (px[:, 1] < y2 - 2)
    if boxes is not None and len(boxes):
        for bx1, by1, bx2, by2 in np.asarray(boxes, dtype=np.float64):
            inside &= ~((px[:, 0] >= bx1) & (px[:, 0] <= bx2) & (px[:, 1] >= by1) & (px[:, 1] <= by2))
    if inside.sum() < 200:
        return None
    pi = np.round(px[inside]).astype(int)
    lab = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).astype(np.int32)
    v = lab[pi[:, 1] - y1, pi[:, 0] - x1]
    g = np.asarray(grass_lab, dtype=np.float64)
    chroma = np.abs(v[:, 1] - 128) + np.abs(v[:, 2] - 128)
    decor = (np.linalg.norm(v - g, axis=1) > 38) & (chroma >= 45) & (v[:, 0] > g[0] - 10)
    strip = np.floor(pts[inside, 0]).astype(int)          # strip k is the yard from k to k + 1
    frac, seen = {}, {}
    for s in np.unique(strip):
        m = strip == s
        seen[int(s)] = int(m.sum())
        frac[int(s)] = float(decor[m].mean())

    def mean_frac(a: int, b: int) -> tuple[float, int]:
        ks = [k for k in range(min(a, b), max(a, b)) if seen.get(k, 0) >= 8]
        if len(ks) < 5:
            return np.nan, 0
        return float(np.mean([frac[k] for k in ks])), len(ks)

    best = None
    for goal in range(int(5 * np.ceil(lo / 5)), int(hi) + 1, 5):
        for side in (1, -1):
            inside_f, n_in = mean_frac(goal, goal + 10 * side)
            outside_f, n_out = mean_frac(goal - 10 * side, goal)
            if n_in < 5 or n_out < 5:
                continue
            if inside_f >= min_inside and outside_f <= max_outside:
                score = inside_f - outside_f
                if best is None or score > best[0]:
                    best = (score, float(goal), side)
    return None if best is None else (best[1], best[2])


def anchor_play(x_rel: np.ndarray, team: np.ndarray, yards_to_goal: float, end_zone: tuple[float, int] | None = None,
                depth: np.ndarray | None = None) -> Anchor | None:
    """Fix the line of scrimmage and the offense's direction for one play.

    `x_rel`, `team`: the players before the snap. `end_zone`: from find_end_zone, if any.
    """
    split = split_teams(x_rel, team)
    if split is None:
        return None
    los_est, low_team, accuracy = split
    if accuracy < 0.7:
        return None
    x_rel, team = np.asarray(x_rel, dtype=np.float64), np.asarray(team)
    direction, source = None, ""
    if end_zone is not None:
        goal, side = end_zone
        attacking = abs(goal - (los_est + side * yards_to_goal))
        backed_up = abs(goal - (los_est - side * (100.0 - yards_to_goal)))
        if min(attacking, backed_up) <= 3.5:
            direction = side if attacking <= backed_up else -side
            source = "end zone"
    if direction is None:
        # the defense plays deeper off the ball than the offense does
        deep = {}
        for t in (0, 1):
            d = np.sort(np.abs(x_rel[team == t] - los_est))
            deep[t] = float(d[-2]) if len(d) >= 2 else 0.0          # second deepest: one stray official can't decide it
        if abs(deep[0] - deep[1]) < 2.0:
            return None
        defense = 0 if deep[0] > deep[1] else 1
        offense_low = low_team != defense
        direction = 1 if offense_low else -1
        source = "formation"
    offense_team = low_team if direction > 0 else 1 - low_team
    # the goal line is a yard line, so the line of scrimmage sits a known distance from one
    goal = 5.0 * np.round((los_est + direction * yards_to_goal) / 5.0)
    los_rel = goal - direction * yards_to_goal
    if abs(los_rel - los_est) > 2.6:
        los_rel = los_est
    return Anchor(los_rel=float(los_rel), direction=int(direction), los_abs=110.0 - float(yards_to_goal),
                  offense_team=int(offense_team), source=source, split_accuracy=float(accuracy))
